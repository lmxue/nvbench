# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p
#

import argparse
import concurrent.futures
import glob
import os
import tempfile

import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from requests import session
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

import scipy

PCS = np.ones(257)      # Perceptual Contrast Stretching
PCS[0:3] = 1
PCS[3:6] = 1.070175439
PCS[6:9] = 1.182456140
PCS[9:12] = 1.287719298
PCS[12:138] = 1.4       # Pre Set
PCS[138:166] = 1.322807018
PCS[166:200] = 1.238596491
PCS[200:241] = 1.161403509
PCS[241:256] = 1.077192982

def Sp_and_phase(signal):
    signal_length = signal.shape[0]
    n_fft = 512
    y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)

    F = librosa.stft(y_pad, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)

    Lp = PCS * np.transpose(np.log1p(np.abs(F)), (1, 0))
    phase = np.angle(F)

    NLp = np.transpose(Lp, (1, 0))

    return NLp, phase, signal_length


def SP_to_wav(mag, phase, signal_length):
    mag = np.expm1(mag)
    Rec = np.multiply(mag, np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming, length=signal_length)
    return result

def pre(audio):
    noisy_LP, Nphase, signal_length = Sp_and_phase(audio)
    enhanced_wav = SP_to_wav(noisy_LP, Nphase, signal_length)
    enhanced_wav = enhanced_wav/np.max(abs(enhanced_wav))

    return enhanced_wav


class ComputeScore:
    def __init__(self, primary_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate, is_personalized_MOS):
        tmp_wav_path = None

        # -------- 新增：mp3 -> 临时 wav，再走原有 sf.read 流程 --------
        try:
            ext = os.path.splitext(fpath)[1].lower()
            if ext == ".mp3":
                # 用 librosa 读取 mp3（依赖环境的 audioread/ffmpeg 支持）
                aud, input_fs = librosa.load(fpath, sr=None, mono=False)
                # aud 可能是 (T,) 或 (C, T)
                if isinstance(aud, np.ndarray) and aud.ndim > 1:
                    aud = aud[0, :]

                # 写成临时 wav（保持原始采样率，不在这里 resample）
                fd, tmp_wav_path = tempfile.mkstemp(prefix="dnsmos_mp3_", suffix=".wav")
                os.close(fd)
                sf.write(tmp_wav_path, aud, input_fs)

                # 然后用原逻辑读 wav
                aud, input_fs = sf.read(tmp_wav_path)
            else:
                aud, input_fs = sf.read(fpath)
        finally:
            # 注意：这里不立刻删 tmp wav，因为如果上面异常，可能还没读完
            # 真正删除放在函数末尾
            pass

        try:
            if len(aud.shape) > 1:
                aud = aud[:, 0]

            fs = sampling_rate
            if input_fs != fs:
                audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
            else:
                audio = aud

            # audio = pre(audio)

            actual_audio_len = len(audio)
            len_samples = int(INPUT_LENGTH*fs)
            while len(audio) < len_samples:
                audio = np.append(audio, audio)

            num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH) + 1
            hop_len_samples = fs
            predicted_mos_sig_seg_raw = []
            predicted_mos_bak_seg_raw = []
            predicted_mos_ovr_seg_raw = []
            predicted_mos_sig_seg = []
            predicted_mos_bak_seg = []
            predicted_mos_ovr_seg = []

            for idx in range(num_hops):
                audio_seg = audio[int(idx*hop_len_samples): int((idx+INPUT_LENGTH)*hop_len_samples)]
                if len(audio_seg) < len_samples:
                    continue

                input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
                oi = {'input_1': input_features}
                mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
                mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                    mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
                )
                predicted_mos_sig_seg_raw.append(mos_sig_raw)
                predicted_mos_bak_seg_raw.append(mos_bak_raw)
                predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
                predicted_mos_sig_seg.append(mos_sig)
                predicted_mos_bak_seg.append(mos_bak)
                predicted_mos_ovr_seg.append(mos_ovr)

            clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr': fs}
            clip_dict['num_hops'] = num_hops
            clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
            clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
            clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
            clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
            clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
            clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
            return clip_dict

        finally:
            # 清理临时 wav
            if tmp_wav_path is not None:
                try:
                    os.remove(tmp_wav_path)
                except OSError:
                    pass


def main(args):
    models = glob.glob(os.path.join(args.testset_dir, "*"))
    audio_clips_list = []

    if args.personalized_MOS:
        primary_model_path = os.path.join('pDNSMOS', 'sig_bak_ovr.onnx')
    else:
        primary_model_path = os.path.join('DNSMOS', 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path)

    rows = []
    clips = []
    # -------- 新增：顶层也把 mp3 加进来 --------
    clips = glob.glob(os.path.join(args.testset_dir, "*.wav")) + glob.glob(os.path.join(args.testset_dir, "*.mp3"))

    is_personalized_eval = args.personalized_MOS
    desired_fs = SAMPLING_RATE
    for m in tqdm(models):
        max_recursion_depth = 10
        audio_path = os.path.join(args.testset_dir, m)
        # -------- 新增：每层搜索 wav + mp3 --------
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav")) + glob.glob(os.path.join(audio_path, "*.mp3"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav")) + glob.glob(os.path.join(audio_path, "*.mp3"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {
            executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip
            for clip in clips
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)

    df = pd.DataFrame(rows)
    if args.csv_path:
        csv_path = args.csv_path
        df.to_csv(csv_path)
    else:
        print(df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', "--testset_dir", default='.',
        help='Path to the dir containing audio clips in .wav/.mp3 to be evaluated'
    )
    parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument(
        '-p', "--personalized_MOS", action='store_true',
        help='Flag to indicate if personalized MOS score is needed or regular'
    )

    args = parser.parse_args()
    main(args)
