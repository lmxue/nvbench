#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import re
import threading
import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import google.generativeai as genai

# ==============================================================================
# 0) Environment / Proxy / API key
# ==============================================================================
def ensure_proxy(default_proxy="http://127.0.0.1:7890"):
    """
    Ensure http(s)_proxy is set (common requirement on many clusters).
    If you already export http_proxy/https_proxy outside, this won't override them.
    """
    if not os.environ.get("http_proxy"):
        os.environ["http_proxy"] = default_proxy
    if not os.environ.get("https_proxy"):
        os.environ["https_proxy"] = default_proxy

    # Uppercase variants for some libs
    if not os.environ.get("HTTP_PROXY"):
        os.environ["HTTP_PROXY"] = os.environ["http_proxy"]
    if not os.environ.get("HTTPS_PROXY"):
        os.environ["HTTPS_PROXY"] = os.environ["https_proxy"]

# If your environment needs a proxy, set USE_PROXY=1 and optionally export
# http_proxy / https_proxy before running. Example:
#   export USE_PROXY=1
#   export http_proxy="http://127.0.0.1:7890"
if os.environ.get("USE_PROXY", "0").strip() in ("1", "true", "yes"):
    ensure_proxy()

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("FATAL: GEMINI_API_KEY is not set. Please export GEMINI_API_KEY first.")

genai.configure(api_key=API_KEY)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-pro")

SYSTEM_INSTRUCTION = """
You are a STRICT professional MOS rater for TTS audio.
Do not inflate scores. Use the full scale. Score 5 is extremely rare.
Follow the rubric and output JSON only (no markdown).
"""

model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION)

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.2"))
N_ROUNDS = 3
GLOBAL_SEED = int(os.environ.get("GLOBAL_SEED", "1234"))

ERROR_LOG_FILE = os.environ.get("ERROR_LOG_FILE", "eval_errors.log")


# ==============================================================================
# 1) Metrics (same as yours)
# ==============================================================================
METRICS_INFO_EN = {
    "Overall Naturalness": "How natural and human-like does the speech sound overall? Focus on whether it feels 'real'. Ignore background noise (rated in Quality).",
    "Overall Quality": "Signal-level audio quality. Rate how clean, clear, and comfortable to listen to the audio signal itself is, ignoring naturalness or content.",
    "Caption–Audio Match (CAM)": "Does the synthesized audio match the caption/prompt in terms of speaker, emotion, and scenario? Ignore detailed NVC timing (rated in IF).",
    "Overall Expression": "Overall expressive effect of speech + NVC, and how well the performance conveys emotion and attitude. Judge by listening alone, not comparing to the caption.",
    "NVC Instruction Following (IF)": "In prompt-based synthesis, does the model follow the NVC-related instructions (type, position) in the text? If no NVC is heard, score 0.",
    "NVC Accuracy": "In tag-based TTS, do the generated NVCs match the input NVC tags (type, position)? If no NVC is heard, score 0.",
    "NVC Perceptual Effect (PE)": "Regardless of instructions, how natural and expressive do the NVCs sound? Do they enhance the performance? If no NVC is heard, score 0.",
}
METRIC_SCORING_RULES_EN = {
    "Overall Naturalness": {
        "5": "Indistinguishable from real human speech.",
        "4": "Sounds like a real person with only slight unnaturalness.",
        "3": "Intelligible, but synthetic qualities are clearly noticeable.",
        "2": "Rather unnatural, sounds more like a machine.",
        "1": "Extremely unnatural or almost intolerable."
    },
    "Overall Quality": {
        "5": "Very clean and clear audio, like a studio recording.",
        "4": "Clear and comfortable audio with only slight, non-intrusive noise/artifacts.",
        "3": "Noticeable issues (noise, compression) that may distract.",
        "2": "Serious quality problems (heavy noise, distortion).",
        "1": "Very poor or unusable audio quality."
    },
    "Caption–Audio Match (CAM)": {
        "5": "Extremely well-matched to caption in voice, emotion, and scene.",
        "4": "Good match with only minor deviations.",
        "3": "Roughly matches, but with at least one obvious core mismatch (e.g., gender, emotion).",
        "2": "Poor match with multiple clear inconsistencies.",
        "1": "Almost no match with the caption."
    },
    "Overall Expression": {
        "5": "Very vivid, with clear emotion and strong dynamics.",
        "4": "Expression is generally good and engaging.",
        "3": "Some emotion is present, but it's relatively flat.",
        "2": "Expression is rather stiff or disjointed.",
        "1": "Almost no perceivable emotion or expressive variation."
    },
    "NVC Instruction Following (IF)": {
        "5": "Fully follows the NVC instruction (type and position).",
        "4": "Generally matches the instruction with minor deviations.",
        "3": "Recognizable NVC, but with clear mismatches in type or timing.",
        "2": "NVC heard, but it does not match most of the instruction.",
        "1": "NVC is almost unrelated to the instruction.",
        "0": "No NVC or almost no NVC is audible."
    },
    "NVC Accuracy": {
        "5": "The NVC matches the tag perfectly (type and timing).",
        "4": "The NVC generally matches the tag well.",
        "3": "The NVC is only partially consistent with the tag.",
        "2": "The NVC mostly does not match the expected tag.",
        "1": "The NVC is almost completely unrelated to the tag.",
        "0": "No NVC or almost no NVC is audible."
    },
    "NVC Perceptual Effect (PE)": {
        "5": "The NVC sounds very natural, expressive, and enhances emotion.",
        "4": "The NVC sounds generally natural and works as a plus.",
        "3": "Naturalness is average, with some synthetic feeling.",
        "2": "The NVC sounds rather stiff, overacted, or connects poorly.",
        "1": "The NVC is extremely unnatural and breaks immersion.",
        "0": "No NVC or almost no NVC is audible."
    },
}

PROMPT_METRICS = [
    "Overall Naturalness",
    "Overall Quality",
    "Caption–Audio Match (CAM)",
    "NVC Instruction Following (IF)",
    "NVC Perceptual Effect (PE)",
]
TAG_METRICS = [
    "Overall Naturalness",
    "Overall Quality",
    "Overall Expression",
    "NVC Accuracy",
    "NVC Perceptual Effect (PE)",
]


# ==============================================================================
# Switch: comparative judging (simulate subjective test)
# ==============================================================================
# Set # Set GROUP_COMPARE=1 to evaluate multiple systems for the SAME (tag, sample_id) together,
# with anonymized labels (A/B/C/...) and relative scoring. This greatly stabilizes ranking.
GROUP_COMPARE = os.environ.get("GROUP_COMPARE", "0").strip().lower() in ("1", "true", "yes", "y", "on")

# How many systems to show per comparison set.
# - If GROUP_SIZE<=0 (default), AUTO: show all systems available under this task (best matches real subjective tests).
# - If GROUP_SIZE>0 and smaller than total systems, the script will split systems into multiple sets.
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "0"))

# Optional: always include a specific system in each comparison set *only when splitting is needed*.
# In real subjective tests you typically DON'T use anchors; leave empty to disable.
ANCHOR_SYSTEM = os.environ.get("ANCHOR_SYSTEM", "").strip()

# Simulate multiple independent human raters (different calibration profiles).
N_RATERS = int(os.environ.get("N_RATERS", "5"))

# Coverage controls how many raters (on average) will score each sample:
# - 1.0 => every rater scores every sample (maximum reliability, highest cost)
# - <1.0 => each sample is scored by a deterministic subset of raters, BUT we guarantee every sample is covered
#          by at least MIN_RATERS_PER_SAMPLE raters (to avoid missing samples).
RATER_COVERAGE = float(os.environ.get("RATER_COVERAGE", "1.0"))

# Ensure each sample is scored by at least this many raters (default 1).
MIN_RATERS_PER_SAMPLE = int(os.environ.get("MIN_RATERS_PER_SAMPLE", "1"))

# Optional: comma-separated list of rater profiles; if empty, use built-ins.
RATER_PROFILES_ENV = os.environ.get("RATER_PROFILES", "").strip()


METRIC_TO_JSON_KEY = {
    "Overall Naturalness": "overall_naturalness",
    "Overall Quality": "overall_quality",
    "Caption–Audio Match (CAM)": "cam",
    "Overall Expression": "overall_expression",
    "NVC Instruction Following (IF)": "nvc_if",
    "NVC Accuracy": "nvc_accuracy",
    "NVC Perceptual Effect (PE)": "nvc_pe",
}

# ==============================================================================
# 2) Hard-cap rules (point #1)
# ==============================================================================
ALLOWED_ISSUES = [
    "clipping", "distortion", "background_noise", "codec_artifacts", "harsh_sibilance",
    "metallic_timbre", "robotic_prosody", "glitches_breaks", "unnatural_phonemes"
]

QUALITY_CAP_ISSUES = {"clipping", "distortion", "background_noise", "codec_artifacts", "harsh_sibilance"}
NATURALNESS_CAP_ISSUES = {"metallic_timbre", "robotic_prosody", "glitches_breaks", "unnatural_phonemes"}

KEYWORD_FALLBACK = {
    # Keyword fallback is only a safety net.
    # Keep terms specific (avoid generic 'artifact/compression/codec/break') to reduce false triggers.
    "clipping": ["clipping", "clipped", "削波", "削顶", "爆音削波"],
    "distortion": ["distortion", "distorted", "overdriven", "失真", "破音", "过载"],
    "background_noise": [
        "background noise", "noise floor", "hiss", "hum", "static",
        "底噪", "电流声", "嗡嗡声", "嘶嘶声", "噪声很大", "环境噪声很大"
    ],
    # Avoid broad 'compression/codec/artifact' — use strong-signal phrases only.
    "codec_artifacts": [
        "heavy compression", "strong compression artifacts", "bitrate artifacts",
        "warbling", "watery", "swirling", "robotic artifacts",
        "严重压缩伪影", "明显压缩伪影", "码率伪影", "水声感", "涟漪感"
    ],
    "harsh_sibilance": ["sibilance", "harsh s", "piercing s", "齿音重", "尖锐齿音", "刺耳齿音"],
    "metallic_timbre": ["metallic", "tinny", "metallic timbre", "金属音", "铁皮音", "很金属"],
    "robotic_prosody": ["robotic", "machine-like", "mechanical", "flat prosody", "机械感", "机器味", "很机器人", "语气很平"],
    # Avoid generic 'break' (e.g., 'break immersion'). Use specific audio glitch terms.
    "glitches_breaks": ["glitch", "dropout", "audio drop", "stutter", "click", "pop", "crackle", "断音", "卡顿", "爆裂声", "噼啪声"],
    # Avoid generic 'phoneme'. Use specific mispronunciation / intelligibility signals.
    "unnatural_phonemes": [
        "mispronunciation", "wrong pronunciation", "garbled", "mumbled", "slurred",
        "错读", "读错", "发音不准", "吐字不清", "口齿不清", "含混", "声调不对", "不太听得清"
    ],
}

def clamp_int(x, lo, hi):
    try:
        xi = int(x)
    except Exception:
        return None
    return max(lo, min(hi, xi))

def extract_issues(pred: dict) -> set:
    """
    Extract issues from model output.
    Priority:
      1) Structured 'issues' field (must be from ALLOWED_ISSUES)
      2) Keyword fallback on ONLY: heard_summary, overall_quality_reason, overall_naturalness_reason
         (avoid scanning CAM/IF/PE/Expression reasons to reduce false triggers)

    NOTE: This function is used only for post-processing consistency on Overall Quality/Naturalness.
    """
    issues = set()
    if not isinstance(pred, dict):
        return issues

    # 1) Structured issues (preferred)
    raw = pred.get("issues", [])
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, str) and it.strip() in ALLOWED_ISSUES:
                issues.add(it.strip())
    elif isinstance(raw, str) and raw.strip() in ALLOWED_ISSUES:
        issues.add(raw.strip())

    # 2) Keyword fallback (safety net) — limited fields to reduce false triggers
    # Negation guard: if a negation appears right before the keyword, do not count it as a hit.
    neg_pat = re.compile(r"\b(no|not|without)\b|没有|无", re.IGNORECASE)

    def contains_kw(text: str, kw: str) -> bool:
        if not text or not kw:
            return False
        t = text.lower()
        k = kw.lower()
        idx = t.find(k)
        if idx < 0:
            return False
        left = t[max(0, idx - 12):idx]
        if neg_pat.search(left):
            return False
        return True

    fields = []
    for k in ("heard_summary", "overall_quality_reason", "overall_naturalness_reason"):
        v = pred.get(k)
        if isinstance(v, str) and v.strip():
            fields.append(v.lower())
    joined = " ".join(fields)

    for issue, kws in KEYWORD_FALLBACK.items():
        for kw in kws:
            if contains_kw(joined, kw):
                issues.add(issue)
                break

    return issues

def apply_hard_caps(pred: dict, task_mode: str) -> dict:
    """
    Apply rubric-consistent caps on Overall Quality / Overall Naturalness.

    Goal:
      - Avoid score collapse at 3 caused by over-eager hard caps.
      - Enforce consistency only when the judge's own text indicates clear problems,
        especially *severe/heavy* problems.

    Three-tier logic (cap is an upper bound; never boosts scores):
      Overall Quality:
        - heavy/serious noise or distortion -> cap <= 2
        - noticeable, distracting issues     -> cap <= 3
        - any stated issue but score==5      -> cap <= 4  (since 5 means no audible issues)

      Overall Naturalness: analogous tiers.

    Notes:
      - Severity is inferred from the judge's wording in:
          heard_summary + overall_quality_reason / overall_naturalness_reason
      - If the judge mentions an issue without any severity cues, we avoid over-capping;
        we only prevent contradictory '5' in that case.
    """
    if not isinstance(pred, dict):
        return pred

    out = dict(pred)
    issues = extract_issues(out)
    caps_applied = {}

    def cap(score_key: str, cap_value: int, lo: int, hi: int, why: str):
        s = clamp_int(out.get(score_key), lo, hi)
        if s is None:
            return
        if s > cap_value:
            out[score_key] = cap_value
            caps_applied[score_key] = why

    def joined_text(*keys: str) -> str:
        parts = []
        for k in keys:
            v = out.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        return " ".join(parts).lower()

    def severity_flags(text: str):
        # Strong/severe signals -> "heavy"
        heavy = bool(re.search(
            r"\b(heavy|strong|severe|serious|extreme)\b|明显|严重|重度|持续|频繁|一直|影响理解|听不清|不舒服|刺耳|难以忍受|hard to listen|uncomfortable|painful|hurts",
            text, flags=re.IGNORECASE
        ))
        # Moderate signals -> "noticeable"
        moderate = bool(re.search(
            r"\b(noticeable|distract|distracting|annoying|bothersome|may distract)\b|干扰|分散注意|明显可闻|听得出|听出来|比较明显",
            text, flags=re.IGNORECASE
        ))
        # Mild signals -> "slight"
        slight = bool(re.search(
            r"\b(slight|minor|subtle|non-intrusive)\b|轻微|不明显|不太影响|基本不影响|偶尔",
            text, flags=re.IGNORECASE
        ))
        return heavy, moderate, slight

    # ----------------------------
    # Overall Quality (1–5)
    # ----------------------------
    quality_issue_any = issues & (set(QUALITY_CAP_ISSUES) | {"glitches_breaks"})
    qt = joined_text("heard_summary", "overall_quality_reason")
    q_heavy, q_mod, q_slight = severity_flags(qt)

    # If any issue is stated but score is 5, it's inconsistent with the rubric definition of 5.
    if quality_issue_any:
        cap("overall_quality_score", 4, 1, 5, "issue_present_so_not_5")

    # Three-tier caps only when severity cues exist (avoid over-capping vague wording)
    if q_heavy and (issues & {"background_noise", "distortion", "clipping"}):
        cap("overall_quality_score", 2, 1, 5, "heavy_noise_or_distortion")
    elif q_mod and quality_issue_any:
        cap("overall_quality_score", 3, 1, 5, "noticeable_distracting_issue")

    # ----------------------------
    # Overall Naturalness (1–5)
    # ----------------------------
    nat_issue_any = issues & set(NATURALNESS_CAP_ISSUES)
    nt = joined_text("heard_summary", "overall_naturalness_reason")
    n_heavy, n_mod, n_slight = severity_flags(nt)

    if nat_issue_any:
        cap("overall_naturalness_score", 4, 1, 5, "issue_present_so_not_5")

    if n_heavy and (issues & {"glitches_breaks", "unnatural_phonemes", "robotic_prosody"}):
        cap("overall_naturalness_score", 2, 1, 5, "severely_unnatural_or_breaks")
    elif n_mod and nat_issue_any:
        cap("overall_naturalness_score", 3, 1, 5, "noticeably_synthetic")

    # Keep original conservative cap for tag mode if NVC totally absent (does NOT affect Quality/Naturalness).
    if task_mode == "tag":
        nvc_acc = clamp_int(out.get("nvc_accuracy_score"), 0, 5)
        nvc_pe = clamp_int(out.get("nvc_pe_score"), 0, 5)
        if nvc_acc == 0 or nvc_pe == 0:
            s = clamp_int(out.get("overall_expression_score"), 1, 5)
            if s is not None and s > 3:
                out["overall_expression_score"] = 3
                caps_applied["overall_expression_score"] = "tag_mode_nvc_absent"

    out["issues"] = sorted(list(issues))
    out["caps_applied"] = caps_applied
    return out

# ==============================================================================
# 3) Prompt builder (point #1, #2)
# ==============================================================================
STRICTNESS_BLOCK = f"""
--- STRICTNESS & CONSISTENCY RULES (CRITICAL) ---
You must be a careful, consistent judge. Use the full 1–5 scale and avoid both inflation and unnecessary harshness.

A) IMPORTANT: Separate Naturalness vs Quality
- Overall Naturalness measures human-likeness (prosody, pronunciation, flow). Do NOT penalize Naturalness for mild background hiss/compression; those belong to Quality.
- Overall Quality measures audio fidelity/cleanliness (noise, distortion/clipping, codec artifacts, clicks/pops/dropouts, harsh sibilance). Do NOT penalize Quality just because the voice sounds synthetic/robotic; that belongs to Naturalness.

B) Score meaning (use these anchors)
Overall Quality:
- 5: Very clean and clear, like a studio recording. No noticeable noise/artifacts/distortion.
- 4: Clear and comfortable; only slight, non-intrusive noise/artifacts.
- 3: Noticeable issues (noise/codec artifacts/etc.) that may distract.
- 2: Serious quality problems (heavy noise/distortion/clipping) that reduce comfort/clarity.
- 1: Very poor or unusable quality.

Overall Naturalness:
- 5: Indistinguishable from real human speech (VERY rare).
- 4: Very natural with minor synthetic cues.
- 3: Noticeably synthetic/robotic traits.
- 2: Strongly synthetic or awkward.
- 1: Very unnatural.

C) Issues (must output)
Output an "issues" list using ONLY labels from this allowed set:
{ALLOWED_ISSUES}
If no issues are present, output [].
Only list issues you can actually hear.

D) Consistency rules you MUST follow
- If you list ANY Quality-related issue (clipping/distortion/background_noise/codec_artifacts/harsh_sibilance/glitches_breaks),
  you MUST NOT give Overall Quality = 5.
  * slight/minor/non-intrusive -> Quality is usually 4
  * noticeable/distracting     -> Quality is usually 3
  * heavy/strong/severe        -> Quality is usually 1–2

- If you list ANY Naturalness-related issue (metallic_timbre/robotic_prosody/unnatural_phonemes/glitches_breaks),
  you MUST NOT give Overall Naturalness = 5.

E) Sanity check
- If you give any metric a 5, the reason must explicitly state: "no noticeable issues".
"""

def build_json_schema(active_metrics):
    # valid JSON example (no trailing commas)
    obj = {
        "heard_summary": "",
        "issues": [],
    }
    for metric in active_metrics:
        key = METRIC_TO_JSON_KEY[metric]
        obj[f"{key}_score"] = 1 if "NVC" not in metric else 0
        obj[f"{key}_reason"] = ""
    obj["caps_applied"] = {}
    return json.dumps(obj, indent=2)

def build_evaluation_prompt(task_type: str) -> str:
    is_prompt_task = "prompt" in task_type
    task_description = "prompt-based" if is_prompt_task else "tag-based"
    # Always evaluate all metrics (ONLY_QN switch removed)
    active_metrics = PROMPT_METRICS if is_prompt_task else TAG_METRICS

    base_prompt = (
        f"You are an expert evaluator for a {task_description} text-to-speech system.\n"
        "You will be given ONE audio sample and its source text.\n"
        "IMPORTANT: The system identity is UNKNOWN and IRRELEVANT. Do not assume any system.\n"
        "Rate the sample on the specified metrics.\n\n"
        "Carefully review the metric definitions and scoring rubric below.\n"
    )

    definitions_block = "--- METRIC DEFINITIONS ---\n"
    for metric in active_metrics:
        definitions_block += f"\n### {metric}\n{METRICS_INFO_EN[metric]}\n"

    rules_block = "\n--- SCORING RUBRIC ---\n"
    for metric in active_metrics:
        rules_block += f"\n#### {metric}\n"
        for score, desc in METRIC_SCORING_RULES_EN[metric].items():
            rules_block += f"- Score {score}: {desc}\n"

    schema = build_json_schema(active_metrics)

    output_instructions = (
        "\n" + STRICTNESS_BLOCK + "\n"
        "\n--- OUTPUT FORMAT ---\n"
        "Return ONLY a valid JSON object matching this schema example (same keys). No extra text.\n"
        f"{schema}\n\n"
        "--- SOURCE TEXT FOR THIS SAMPLE ---\n"
    )
    return base_prompt + definitions_block + rules_block + output_instructions


# ==============================================================================
# ==============================================================================
# 3.5) Comparative judging utilities (GROUP_COMPARE)
# ==============================================================================
def _parse_list_env(s: str):
    if not s:
        return []
    parts = re.split(r"[,\s]+", s.strip())
    return [p for p in parts if p]

def get_rater_profiles():
    """
    Return a list of (profile_name, profile_instruction) used to simulate different human raters.
    If RATER_PROFILES env is provided, we treat each entry as a profile_name and use a generic instruction.
    """
    if RATER_PROFILES_ENV:
        names = _parse_list_env(RATER_PROFILES_ENV)
        profiles = []
        for nm in names:
            profiles.append((nm, f"You are a rater with profile '{nm}'. Apply your own consistent calibration."))
        return profiles

    # Built-in profiles (calibration differences). Keep them small to avoid affecting other metrics.
    return [
        ("balanced", "You are a balanced rater. Use rubric faithfully; 3=noticeable, 4=slight, 5=almost perfect."),
        ("quality_strict", "You are highly sensitive to audio artifacts (noise/distortion/compression). Use lower Quality when artifacts are noticeable."),
        ("naturalness_strict", "You are highly sensitive to prosody/pronunciation/robotic cues. Use lower Naturalness when synthetic traits are noticeable."),
        ("lenient", "You are slightly lenient. If issues are only slight/non-intrusive, prefer 4 over 3."),
        ("harsh", "You are slightly harsh. If you notice any issue beyond very subtle, prefer 3 over 4."),
    ]

def _label_sequence(n: int):
    letters = [chr(ord("A") + i) for i in range(26)]
    if n <= 26:
        return letters[:n]
    # Fallback: A1, A2...
    out = []
    i = 0
    while len(out) < n:
        base = letters[i % 26]
        k = i // 26
        out.append(f"{base}{k+1}")
        i += 1
    return out

def build_group_compare_schema(labels, task_type: str):
    """
    Schema: per-label scoring. We keep fields close to single-sample judge to reuse cap logic.
    Includes all metrics based on task type (prompt or tag).
    Returns a JSON string example (for prompt display).
    """
    is_prompt_task = "prompt" in task_type
    active_metrics = PROMPT_METRICS if is_prompt_task else TAG_METRICS
    
    results_obj = {}
    for lb in labels:
        result_item = {
            "heard_summary": "",
            "issues": [],
        }
        # Add all metrics for this task type
        for metric in active_metrics:
            key = METRIC_TO_JSON_KEY[metric]
            result_item[f"{key}_score"] = 1 if "NVC" not in metric else 0
            result_item[f"{key}_reason"] = ""
        results_obj[lb] = result_item
    
    # Build ranking dict for all metrics
    ranking_dict = {}
    for metric in active_metrics:
        key = METRIC_TO_JSON_KEY[metric]
        ranking_dict[key] = labels
    
    schema = {
        "results": results_obj,
        "ranking": ranking_dict,
        "notes": ""
    }
    return json.dumps(schema, ensure_ascii=False, indent=2)

def build_group_compare_json_schema(labels, task_type: str):
    """
    Build a JSON Schema (Draft 7) for Gemini API response_schema parameter.
    This enforces that the model returns all required metrics.
    """
    is_prompt_task = "prompt" in task_type
    active_metrics = PROMPT_METRICS if is_prompt_task else TAG_METRICS
    
    # Build properties for each label's result
    label_result_properties = {
        "heard_summary": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
    }
    label_result_required = ["heard_summary", "issues"]
    
    # Add all metrics
    for metric in active_metrics:
        key = METRIC_TO_JSON_KEY[metric]
        label_result_properties[f"{key}_score"] = {"type": "integer"}
        label_result_properties[f"{key}_reason"] = {"type": "string"}
        label_result_required.extend([f"{key}_score", f"{key}_reason"])
    
    # Build results object schema
    results_properties = {}
    for lb in labels:
        results_properties[lb] = {
            "type": "object",
            "properties": label_result_properties,
            "required": label_result_required,
            # Allow additional properties (e.g., caps_applied) but require all metrics
        }
    
    # Build ranking object schema
    ranking_properties = {}
    for metric in active_metrics:
        key = METRIC_TO_JSON_KEY[metric]
        ranking_properties[key] = {
            "type": "array",
            "items": {"type": "string", "enum": labels}
        }
    
    # Complete JSON Schema
    json_schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "object",
                "properties": results_properties,
                "required": labels,
            },
            "ranking": {
                "type": "object",
                "properties": ranking_properties,
                "required": [METRIC_TO_JSON_KEY[m] for m in active_metrics],
            },
            "notes": {"type": "string"}
        },
        "required": ["results", "ranking", "notes"],
    }
    
    return json_schema

def build_group_compare_prompt(task_type: str, labels, rater_profile_name: str, rater_profile_inst: str, rater_id: int):
    """
    Comparative prompt: present multiple anonymized audios for SAME source text.
    Output per-label scores for all metrics + rankings.
    """
    task_desc = "prompt-based" if "prompt" in task_type else "tag-based"
    is_prompt_task = "prompt" in task_type
    active_metrics = PROMPT_METRICS if is_prompt_task else TAG_METRICS
    schema = build_group_compare_schema(labels, task_type)

    # Build metric definitions and scoring rules
    definitions_block = "--- METRIC DEFINITIONS ---\n"
    for metric in active_metrics:
        definitions_block += f"\n### {metric}\n{METRICS_INFO_EN[metric]}\n"

    rules_block = "\n--- SCORING RUBRIC ---\n"
    for metric in active_metrics:
        rules_block += f"\n#### {metric}\n"
        for score, desc in METRIC_SCORING_RULES_EN[metric].items():
            rules_block += f"- Score {score}: {desc}\n"

    # Build ranking instructions
    ranking_instructions = ""
    for metric in active_metrics:
        key = METRIC_TO_JSON_KEY[metric]
        ranking_instructions += f"- ranking.{key}: labels sorted from best to worst (ties allowed by placing adjacent; do not repeat labels)\n"

    # Build task instructions listing all metrics
    task_instructions = "TASK:\n"
    task_instructions += f"For EACH candidate (labelled {', '.join(labels)}), listen fully and rate on ALL of the following metrics:\n"
    for idx, metric in enumerate(active_metrics, 1):
        key = METRIC_TO_JSON_KEY[metric]
        if metric == "Overall Naturalness":
            task_instructions += f"{idx}) {metric} (1–5): human-likeness (prosody/pronunciation/flow). Ignore mild background noise (Quality covers it).\n"
        elif metric == "Overall Quality":
            task_instructions += f"{idx}) {metric} (1–5): signal fidelity/cleanliness (noise, compression artifacts, distortion, clipping, harsh sibilance). Ignore robotic prosody (Naturalness covers it).\n"
        else:
            # For other metrics, use a brief description
            min_score = "0" if "NVC" in metric else "1"
            task_instructions += f"{idx}) {metric} ({min_score}–5): {METRICS_INFO_EN[metric]}\n"

    # IMPORTANT: do not mention system names; only labels.
    prompt = f"""
You are an expert evaluator for {task_desc} text-to-speech.

You will be given ONE source text and MULTIPLE candidate audios that all attempt to synthesize the SAME content.
The system identity is UNKNOWN and MUST NOT be inferred.
Treat each candidate independently but calibrate scores RELATIVELY within this set (like a real listening test).

RATER SIMULATION:
- You are simulated rater #{rater_id} with profile: {rater_profile_name}.
- {rater_profile_inst}

{definitions_block}

{rules_block}

{task_instructions}

KEY CONSISTENCY RULES:
- Use the full scale. Most fall in 2–4; 5 is rare and ONLY if you notice no artifacts AND no synthetic cues.
- If you notice ANY artifact that affects Quality, do NOT give Quality=5.
- If you notice ANY synthetic cue that affects Naturalness, do NOT give Naturalness=5.
- If issues are only slight/non-intrusive, prefer 4 over 3.
- Reserve 3 for noticeable/distracting problems; 2 for heavy/serious problems.

ISSUES LIST:
For each candidate, output "issues" as a list using ONLY these labels:
{ALLOWED_ISSUES}
If no issues, output [].

OUTPUT:
Return ONLY one JSON object matching this schema (same keys). No extra text.

Schema example:
{schema}

You must also fill rankings for all metrics:
{ranking_instructions}

SOURCE TEXT:
""".strip()
    return prompt

def _stable_seed_int(*parts):
    """Deterministic 32-bit seed derived from parts (stable across runs)."""
    import hashlib
    s = "||".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).digest()
    v = int.from_bytes(h[:4], "big")  # 32-bit
    return (v ^ GLOBAL_SEED) & 0xFFFFFFFF

def select_raters_for_group(task_name: str, group_key: str, group_idx: int, n_raters: int,
                            coverage: float, min_raters_per_sample: int):
    """
    Deterministically choose which raters will score this (tag, sample_id, set_idx).

    Guarantees:
      - At least `min_raters_per_sample` raters will be selected (>=1).
      - If coverage==1.0, all raters are selected.

    Interpretation of `coverage`:
      - Roughly the probability that a non-mandatory rater is included for a sample.
      - Expected raters per sample ≈ min_raters_per_sample + (n_raters - min_raters_per_sample) * coverage
    """
    n_raters = max(1, int(n_raters))
    coverage = max(0.0, min(1.0, float(coverage)))
    min_k = max(1, min(int(min_raters_per_sample), n_raters))

    if coverage >= 1.0 or n_raters == 1:
        return list(range(n_raters))

    # Mandatory raters: pick `min_k` distinct raters using a stable permutation seeded by group id.
    base_seed = _stable_seed_int(task_name, group_key, group_idx, "mandatory")
    rng = random.Random(base_seed)
    perm = list(range(n_raters))
    rng.shuffle(perm)
    chosen = set(perm[:min_k])

    # Probabilistically include additional raters
    for rid in range(n_raters):
        if rid in chosen:
            continue
        rs = _stable_seed_int(task_name, group_key, group_idx, rid, "coverage")
        if random.Random(rs).random() < coverage:
            chosen.add(rid)

    return sorted(chosen)

def collect_group_candidates(manifest_data, source_data_by_id, task_mode: str, systems_to_evaluate):
    """
    Build mapping:
      group_key = f"{tag}::{sample_id}"
      -> { "tag": tag, "sample_id": sample_id, "source_text": ..., "systems": {system_key: audio_path} }
    """
    by_tag = manifest_data.get("by_tag", {})
    sys_keys = [x.get("key") for x in systems_to_evaluate if x.get("key")]
    sys_key_set = set(sys_keys)

    out = {}
    source_text_field = "caption_with_nvb" if task_mode == "prompt" else "text_with_mark"

    for tag, tag_data in by_tag.items():
        sys_map = tag_data.get("systems", {}) or {}
        # union over systems available under this tag
        for system_key, sys_info in sys_map.items():
            if sys_key_set and system_key not in sys_key_set:
                continue
            paths = (sys_info.get("paths", {}) or {})
            for sample_id, audio_path in paths.items():
                if not isinstance(audio_path, str) or not audio_path.strip():
                    continue
                src = source_data_by_id.get(sample_id)
                if not src:
                    continue
                group_key = f"{tag}::{sample_id}"
                if group_key not in out:
                    out[group_key] = {
                        "tag": tag,
                        "sample_id": sample_id,
                        "source_text": src.get(source_text_field, "N/A"),
                        "systems": {},
                    }
                out[group_key]["systems"][system_key] = audio_path
    return out

def split_systems_into_sets(system_keys, group_size: int, anchor: str, seed_int: int):
    """
    Return a list of system_key lists (each is one comparison set).
    If group_size <= 0, return [all].
    If anchor is set and exists, include it in every set to stabilize scale.
    """
    keys = list(system_keys)
    keys = [k for k in keys if k]
    keys = sorted(set(keys))
    if not keys:
        return []

    if group_size <= 0 or group_size >= len(keys):
        return [keys]

    rng = random.Random(seed_int)
    if anchor and anchor in keys and group_size >= 2:
        others = [k for k in keys if k != anchor]
        rng.shuffle(others)
        chunk = group_size - 1
        sets = []
        for i in range(0, len(others), chunk):
            cur = [anchor] + others[i:i+chunk]
            sets.append(cur)
        return sets

    rng.shuffle(keys)
    sets = []
    for i in range(0, len(keys), group_size):
        sets.append(keys[i:i+group_size])
    return sets

def evaluate_groupcompare_once(task_name: str, task_mode: str, group_key: str, group_idx: int, rater_id: int,
                              rater_profile_name: str, rater_profile_inst: str,
                              systems_in_set, source_text: str, system_to_audio: dict,
                              results, results_lock, output_json: str, error_log_lock):
    """
    Evaluate one comparison set for one rater.
    Writes result into results["items"][item_key] and flushes.
    """
    item_key = f"{group_key}::g{group_idx}::r{rater_id}"
    with results_lock:
        if item_key in results["items"]:
            return

    # Prepare label mapping (deterministic)
    seed_int = _stable_seed_int(task_name, group_key, group_idx, rater_id, ",".join(sorted(systems_in_set)))
    rng = random.Random(seed_int)
    sys_list = list(systems_in_set)
    rng.shuffle(sys_list)

    labels = _label_sequence(len(sys_list))
    label_to_system = {labels[i]: sys_list[i] for i in range(len(sys_list))}
    label_to_audio = {lb: system_to_audio[label_to_system[lb]] for lb in labels}

    # Build prompt
    prompt = build_group_compare_prompt(task_name, labels, rater_profile_name, rater_profile_inst, rater_id)
    eval_prompt_for_item = prompt + f"\n{source_text}\n"

    # Build JSON schema for response validation (enforces all metrics)
    response_schema = build_group_compare_json_schema(labels, task_name)

    # Upload audios and call model
    uploaded_files = []
    try:
        parts = [eval_prompt_for_item]
        # Interleave label markers + audio
        for lb in labels:
            audio_path = label_to_audio[lb]
            audio_file = genai.upload_file(path=audio_path)
            uploaded_files.append(audio_file)
            parts.append(f"\nAUDIO {lb}:\n")
            parts.append(audio_file)

        resp = call_with_retry(
            parts,
            gen_config={
                "temperature": TEMPERATURE,
                "response_mime_type": "application/json",
                "response_schema": response_schema
            },
            max_retries=4,
        )
        parsed = robust_json_load(resp.text)
        # Validate
        if not isinstance(parsed, dict) or "results" not in parsed or not isinstance(parsed["results"], dict):
            raise ValueError("Invalid group-compare JSON: missing results dict")

        # Apply caps per label (only Q/N fields)
        capped = copy.deepcopy(parsed)
        for lb, pred_one in (parsed.get("results") or {}).items():
            if not isinstance(pred_one, dict):
                continue
            pred_one_capped = apply_hard_caps(pred_one, task_mode=task_mode)
            capped.setdefault("results", {})[lb] = pred_one_capped

        # Store
        with results_lock:
            results["items"][item_key] = {
                "status": "ok",
                "task_name": task_name,
                "task_mode": task_mode,
                "group_key": group_key,
                "group_idx": group_idx,
                "rater_id": rater_id,
                "rater_profile": rater_profile_name,
                "ground_truth": {
                    "sample_id": group_key.split("::", 1)[1] if "::" in group_key else group_key,
                    "tag": group_key.split("::", 1)[0] if "::" in group_key else None,
                    "source_text": source_text,
                    "label_to_system": label_to_system,  # bookkeeping only
                    "systems": list(systems_in_set),
                },
                "prediction": parsed,
                "prediction_capped": capped,
                "judge_temperature": TEMPERATURE,
            }
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    except Exception as e:
        write_error_log(group_key, f"GROUPCOMPARE({task_name})", repr(e), lock=error_log_lock)
        with results_lock:
            results["items"][item_key] = {
                "status": "error",
                "error": repr(e),
                "task_name": task_name,
                "group_key": group_key,
                "group_idx": group_idx,
                "rater_id": rater_id,
                "rater_profile": rater_profile_name,
            }
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    finally:
        # cleanup uploaded files
        for uf in uploaded_files:
            try:
                genai.delete_file(uf.name)
            except Exception as e:
                write_error_log(group_key, f"GROUPCOMPARE({task_name})", f"delete_file error: {repr(e)}", lock=error_log_lock)

def generate_summary_report_groupcompare(output_json: str):
    """
    Summarize per-system mean scores and pairwise win-rate from a GROUPCOMPARE result file.
    """
    if not os.path.exists(output_json):
        print(f"[GROUPCOMPARE] file not found: {output_json}")
        return
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", {}) if isinstance(data, dict) else {}
    if not isinstance(items, dict) or not items:
        print("[GROUPCOMPARE] No items.")
        return

    # Aggregate
    score_sum = defaultdict(lambda: defaultdict(float))
    score_n = defaultdict(lambda: defaultdict(int))
    wins = defaultdict(lambda: defaultdict(int))
    comps = defaultdict(lambda: defaultdict(int))

    def add_pairwise(metric_key: str, sys_scores: dict):
        syss = list(sys_scores.keys())
        for i in range(len(syss)):
            for j in range(i+1, len(syss)):
                a, b = syss[i], syss[j]
                sa, sb = sys_scores[a], sys_scores[b]
                comps[metric_key][a] += 1
                comps[metric_key][b] += 1
                if sa > sb:
                    wins[metric_key][a] += 1
                elif sb > sa:
                    wins[metric_key][b] += 1
                # ties: no win

    # Collect all metric keys from results
    all_metric_keys = set()
    for rec in items.values():
        if rec.get("status") != "ok":
            continue
        pred = (rec.get("prediction_capped") or rec.get("prediction") or {})
        results = pred.get("results", {}) if isinstance(pred, dict) else {}
        if not isinstance(results, dict):
            continue
        for lb, one in results.items():
            if isinstance(one, dict):
                for k in one.keys():
                    if k.endswith("_score"):
                        all_metric_keys.add(k)
        break  # Just need to see one record to get all metric keys

    for rec in items.values():
        if rec.get("status") != "ok":
            continue
        gt = rec.get("ground_truth", {})
        label_to_system = gt.get("label_to_system", {}) or {}
        pred = (rec.get("prediction_capped") or rec.get("prediction") or {})
        results = pred.get("results", {}) if isinstance(pred, dict) else {}
        if not isinstance(results, dict):
            continue

        # Collect scores for all metrics
        sys_scores_by_metric = {mk: {} for mk in all_metric_keys}
        for lb, one in results.items():
            sys = label_to_system.get(lb)
            if not sys:
                continue
            for metric_key in all_metric_keys:
                score_val = one.get(metric_key)
                if isinstance(score_val, (int, float)):
                    score_sum[metric_key][sys] += float(score_val)
                    score_n[metric_key][sys] += 1
                    sys_scores_by_metric[metric_key][sys] = float(score_val)

        # Add pairwise comparisons for all metrics
        for metric_key in all_metric_keys:
            sys_scores = sys_scores_by_metric[metric_key]
            if len(sys_scores) >= 2:
                add_pairwise(metric_key, sys_scores)

    # Collect all systems that have scores
    all_systems = set()
    for metric_key in all_metric_keys:
        all_systems.update(score_n[metric_key].keys())
    systems = sorted(all_systems)
    
    if not systems:
        print("[GROUPCOMPARE] No valid scores.")
        return

    def mean(metric_key, sys):
        n = score_n[metric_key].get(sys, 0)
        return (score_sum[metric_key][sys] / n) if n else None

    def winrate(metric_key, sys):
        c = comps[metric_key].get(sys, 0)
        return (wins[metric_key].get(sys, 0) / c) if c else None

    # Rank by mean quality, then naturalness (if available), then other metrics
    def rank_key(s):
        keys = []
        if "overall_quality_score" in all_metric_keys:
            keys.append(-(mean("overall_quality_score", s) or -1e9))
        if "overall_naturalness_score" in all_metric_keys:
            keys.append(-(mean("overall_naturalness_score", s) or -1e9))
        # Add other metrics
        for mk in sorted(all_metric_keys):
            if mk not in ("overall_quality_score", "overall_naturalness_score"):
                keys.append(-(mean(mk, s) or -1e9))
        keys.append(s)  # System name as tiebreaker
        return tuple(keys)
    
    ranked = sorted(systems, key=rank_key)

    print(f"\n{'='*20} GROUPCOMPARE SUMMARY ({os.path.basename(output_json)}) {'='*20}")
    for s in ranked:
        parts = [f"{s:20s}"]
        for metric_key in sorted(all_metric_keys):
            m = mean(metric_key, s)
            w = winrate(metric_key, s)
            n = score_n[metric_key].get(s, 0)
            metric_name = metric_key.replace("_score", "").replace("_", " ").title()
            parts.append(f"{metric_name} {m:.2f} (n={n}) win% {w*100:.1f}%")
        print(" | ".join(parts))
    print(f"{'='*78}\n")

# 4) Data helpers
# ==============================================================================
def load_source_data(file_path: str) -> dict:
    data_by_id = {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        sid = sample.get("id")
        if sid:
            data_by_id[sid] = sample
    return data_by_id

def extract_evaluation_items(manifest, system_key, source_data_by_id, task_mode):
    """Parses manifest, finds corresponding source text, and creates a flat list of items.
       If audio_path is null/empty/non-string, it will be ignored.
    """
    items = []
    if "by_tag" not in manifest:
        return items

    for tag, tag_data in manifest["by_tag"].items():
        sys_map = tag_data.get("systems", {})
        if system_key not in sys_map:
            continue

        system_paths = (sys_map[system_key].get("paths", {}) or {})
        for sample_id, audio_path in system_paths.items():
            # ✅ ignore null/empty/invalid paths
            if not isinstance(audio_path, str) or not audio_path.strip():
                continue

            source_info = source_data_by_id.get(sample_id)
            if not source_info:
                continue

            source_text_field = "caption_with_nvb" if task_mode == "prompt" else "text_with_mark"
            items.append({
                "audio_path": audio_path,
                "sample_id": sample_id,
                "system": system_key,  # not exposed to judge prompt
                "tag": tag,            # not exposed to judge prompt
                "source_text": source_info.get(source_text_field, "N/A"),
            })

    return items

def write_error_log(audio_path: str, system_name: str, error: str, lock=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} - Audio: {audio_path} - System: {system_name} - ERROR: {error}\n"
    if lock:
        with lock:
            with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
    else:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)

def robust_json_load(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Invalid JSON output")

def call_with_retry(parts, gen_config, max_retries=4):
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return model.generate_content(parts, generation_config=gen_config)
        except Exception as e:
            msg = str(e).lower()
            transient = ("429" in msg) or ("resource exhausted" in msg) or ("timeout" in msg) or ("temporarily" in msg) or ("rate" in msg)
            if attempt == max_retries - 1 or not transient:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)

def make_three_folds(items, seed: int):
    rng = random.Random(seed)
    xs = list(items)
    rng.shuffle(xs)
    folds = [[], [], []]
    for i, it in enumerate(xs):
        folds[i % 3].append(it)
    return folds

def generate_summary_report(output_files):
    print(f"\n{'='*25} EVALUATION SUMMARY {'='*25}")
    all_scores = defaultdict(lambda: defaultdict(list))
    any_valid = False
    processed = 0

    for fp in output_files:
        if not os.path.exists(fp):
            continue
        if "__GROUPCOMPARE" in os.path.basename(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed += 1

        parts = os.path.basename(fp).replace("eval_results_", "").replace(".json", "").split("_")
        task_name = parts[0]
        system_key = "_".join(parts[1:])
        system_id = f"{system_key} ({task_name})"

        items = data.get("items", {})
        if not isinstance(items, dict) or not items:
            continue

        for _, rec in items.items():
            pred = rec.get("prediction_capped") or rec.get("prediction") or {}
            for k, v in pred.items():
                if k.endswith("_score") and isinstance(v, (int, float)):
                    any_valid = True
                    all_scores[system_id][k].append(float(v))

    if processed == 0:
        return

    if not any_valid:
        print(f"No valid results found. (Likely all requests failed; check {ERROR_LOG_FILE})")
        return

    for system_id in sorted(all_scores.keys()):
        print(f"\n--- System: {system_id} ---")
        for metric_key in sorted(all_scores[system_id].keys()):
            xs = all_scores[system_id][metric_key]
            if xs:
                print(f"  - {metric_key:<28}: {sum(xs)/len(xs):.2f}  (n={len(xs)})")
        print("-" * (len(system_id) + 12))
    print(f"{'='*68}\n")


# ==============================================================================
# 5) Paths
# ==============================================================================
# Set DATA_DIR to the directory containing your samples_*.json and manifest_*.json files.
# Defaults to ./example_data (for quick testing with the provided example).
SOURCE_DATA_BASE = os.environ.get("DATA_DIR", "./example_data")
SOURCE_DATA_FILES = {
    "en": os.path.join(SOURCE_DATA_BASE, "sampled_en.json"),
    "zh": os.path.join(SOURCE_DATA_BASE, "sampled_zh.json"),
}
MANIFESTS = {
    "zh-prompt": os.path.join(SOURCE_DATA_BASE, "manifest_zh-prompt.json"),
    "zh-tag":    os.path.join(SOURCE_DATA_BASE, "manifest_zh-tag.json"),
    "en-prompt": os.path.join(SOURCE_DATA_BASE, "manifest_en-prompt.json"),
    "en-tag":    os.path.join(SOURCE_DATA_BASE, "manifest_en-tag.json"),
}

# Optional: limit tasks via env, e.g. TASKS="zh-prompt en-tag"
TASKS_FILTER = [t for t in os.environ.get("TASKS", "").split() if t.strip()]

generated_output_files = []


# ==============================================================================
# 6) Main
# ==============================================================================
def main():
    random.seed(GLOBAL_SEED)

    for task_name, manifest_path in ([(k, MANIFESTS[k]) for k in TASKS_FILTER] if TASKS_FILTER else MANIFESTS.items()):
        print(f"\n{'='*20} Starting Task: {task_name} {'='*20}")

        if not os.path.exists(manifest_path):
            print(f"❌ manifest not found: {manifest_path}")
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        lang_key = "zh" if "zh-" in task_name else "en"
        source_data_path = SOURCE_DATA_FILES.get(lang_key, "")
        if not source_data_path or not os.path.exists(source_data_path):
            print(f"❌ source data not found: {source_data_path}")
            continue

        source_data_by_id = load_source_data(source_data_path)

        systems_to_evaluate = manifest_data.get("summary", {}).get("systems", [])
        if not systems_to_evaluate:
            print(f"⚠️ No systems found in manifest for {task_name}")
            continue

        task_mode = "prompt" if "prompt" in task_name else "tag"
        # ----------------------------------------------------------------------
        # GROUP_COMPARE mode: evaluate multiple systems together per (tag, sample_id)
        # ----------------------------------------------------------------------
        if GROUP_COMPARE:
            print(f"🧪 GROUP_COMPARE=1: comparative judging enabled (simulate subjective test).")
            output_json = f"eval_results_{task_name}__GROUPCOMPARE.json"
            if output_json not in generated_output_files:
                generated_output_files.append(output_json)

            # resume
            results = {"__meta__": {}, "items": {}, "__summary__": {}}
            if os.path.exists(output_json):
                try:
                    with open(output_json, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict) and "items" in loaded:
                        results = loaded
                except Exception:
                    pass

            # Build candidate pools
            pools = collect_group_candidates(manifest_data, source_data_by_id, task_mode, systems_to_evaluate)
            if not pools:
                print("⚠️ No pools found for GROUP_COMPARE.")
                continue

            # Build evaluation jobs (each job = one comparison set for one rater)
            rater_profiles = get_rater_profiles()
            if N_RATERS <= 0:
                N_RATERS_eff = 1
            else:
                N_RATERS_eff = N_RATERS

            jobs = []
            for group_key, info in pools.items():
                sys_to_audio = info.get("systems", {}) or {}
                # Require at least 2 systems to compare
                if len(sys_to_audio) < 2:
                    continue

                sys_keys = list(sys_to_audio.keys())
                set_seed = _stable_seed_int(task_name, group_key, "sets")
                sets = split_systems_into_sets(sys_keys, group_size=GROUP_SIZE, anchor=ANCHOR_SYSTEM, seed_int=set_seed)
                if not sets:
                    continue

                for gidx, sys_set in enumerate(sets):
                    # if anchor set but missing in this pool, split_systems_into_sets won't include it
                    if len(sys_set) < 2:
                        continue
                    selected_raters = select_raters_for_group(
                        task_name=task_name,
                        group_key=group_key,
                        group_idx=gidx,
                        n_raters=N_RATERS_eff,
                        coverage=RATER_COVERAGE,
                        min_raters_per_sample=MIN_RATERS_PER_SAMPLE,
                    )
                    for rid in selected_raters:
                        prof_name, prof_inst = rater_profiles[rid % len(rater_profiles)]
                        jobs.append({
                            "group_key": group_key,
                            "group_idx": gidx,
                            "rater_id": rid,
                            "rater_profile_name": prof_name,
                            "rater_profile_inst": prof_inst,
                            "systems_in_set": sys_set,
                            "source_text": info.get("source_text", "N/A"),
                            "system_to_audio": sys_to_audio,
                        })

            if not jobs:
                print("⚠️ GROUP_COMPARE: no valid jobs (need >=2 systems per sample).")
                continue

            fold_seed = (_stable_seed_int(task_name, "GROUPCOMPARE") ^ GLOBAL_SEED) & 0xFFFFFFFF
            folds = make_three_folds(jobs, seed=fold_seed)

            results["__meta__"] = {
                "task_name": task_name,
                "task_mode": task_mode,
                "mode": "GROUPCOMPARE",
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "n_rounds": N_ROUNDS,
                "global_seed": GLOBAL_SEED,
                "fold_seed": fold_seed,
                "group_size": GROUP_SIZE,
                "anchor_system": ANCHOR_SYSTEM,
                "n_raters": N_RATERS_eff,
                "rater_profiles": [p[0] for p in rater_profiles],
                "proxy": {"http_proxy": os.environ.get("http_proxy", ""), "https_proxy": os.environ.get("https_proxy", "")},
                "note": "system identity is NOT exposed; labels are randomized per rater",
            }

            results_lock = threading.Lock()
            error_log_lock = threading.Lock()

            def process_job(job, round_idx):
                evaluate_groupcompare_once(
                    task_name=task_name,
                    task_mode=task_mode,
                    group_key=job["group_key"],
                    group_idx=job["group_idx"],
                    rater_id=job["rater_id"],
                    rater_profile_name=job["rater_profile_name"],
                    rater_profile_inst=job["rater_profile_inst"],
                    systems_in_set=job["systems_in_set"],
                    source_text=job["source_text"],
                    system_to_audio=job["system_to_audio"],
                    results=results,
                    results_lock=results_lock,
                    output_json=output_json,
                    error_log_lock=error_log_lock,
                )

            # rounds on jobs
            for r in range(N_ROUNDS):
                round_queue = folds[r] if r < len(folds) else []
                rng = random.Random((fold_seed + r * 1000003) & 0xFFFFFFFF)
                rng.shuffle(round_queue)

                with results_lock:
                    done = set(results["items"].keys())
                # Filter already-done
                def _job_key(j):
                    return f"{j['group_key']}::g{j['group_idx']}::r{j['rater_id']}"
                round_queue = [j for j in round_queue if _job_key(j) not in done]

                if not round_queue:
                    print(f"ℹ️ Round {r+1}/{N_ROUNDS}: no new GROUP_COMPARE jobs.")
                    continue

                print(f"▶ Round {r+1}/{N_ROUNDS}: GROUP_COMPARE evaluating {len(round_queue)} jobs (~1/3)")

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futures = [ex.submit(process_job, j, r) for j in round_queue]
                    for fu in tqdm(as_completed(futures), total=len(futures), desc=f"GROUPCOMPARE {task_name} R{r+1}"):
                        fu.result()

            # Summary to console
            generate_summary_report_groupcompare(output_json)
            print(f"✅ GROUP_COMPARE done for {task_name}. Results saved to {output_json}")
            continue
        main_eval_prompt = build_evaluation_prompt(task_name)

        for system_info in systems_to_evaluate:
            system_key = system_info.get("key")
            system_name = system_info.get("name", system_key)
            if not system_key:
                continue

            print(f"\n----- Evaluating System: {system_name} ({task_name}) -----")

            output_json = f"eval_results_{task_name}_{system_key}.json"
            missing_log = f"NVS_missing_audio_{task_name}_{system_key}.log"
            open(missing_log, "w", encoding="utf-8").close()

            if output_json not in generated_output_files:
                generated_output_files.append(output_json)

            # resume
            results = {"__meta__": {}, "items": {}, "__summary__": {}}
            if os.path.exists(output_json):
                try:
                    with open(output_json, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict) and "items" in loaded:
                        results = loaded
                    elif isinstance(loaded, dict):
                        results = {"__meta__": {}, "items": loaded, "__summary__": {}}
                except Exception:
                    pass

            full_queue = extract_evaluation_items(manifest_data, system_key, source_data_by_id, task_mode)
            if not full_queue:
                print("ℹ️ No evaluation items found.")
                continue

            # 3 folds (~1/3 each) + per-round shuffle (point #2, #3)
            fold_seed = (hash(task_name + "||" + system_key) ^ GLOBAL_SEED) & 0xFFFFFFFF
            folds = make_three_folds(full_queue, seed=fold_seed)

            results["__meta__"] = {
                "task_name": task_name,
                "task_mode": task_mode,
                "system_key": system_key,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "n_rounds": N_ROUNDS,
                "global_seed": GLOBAL_SEED,
                "fold_seed": fold_seed,
                "proxy": {"http_proxy": os.environ.get("http_proxy", ""), "https_proxy": os.environ.get("https_proxy", "")},
                "note": "system identity is NOT exposed to judge prompt",
            }

            results_lock = threading.Lock()
            error_log_lock = threading.Lock()

            # counters for quick sanity check
            counters = {"ok": 0, "missing": 0, "error": 0}

            def process_item(item, round_idx):
                audio_path = item["audio_path"]
                
                if not isinstance(audio_path, str) or not audio_path.strip():
                    return

                try:
                    # resume skip
                    with results_lock:
                        if audio_path in results["items"]:
                            return

                    if not os.path.exists(audio_path):
                        with error_log_lock:
                            with open(missing_log, "a", encoding="utf-8") as logf:
                                logf.write(f"{audio_path}\n")
                        with results_lock:
                            results["items"][audio_path] = {
                                "round": round_idx,
                                "status": "missing",
                                "ground_truth": {"sample_id": item["sample_id"], "tag": item["tag"], "system": item["system"]},
                            }
                        return

                    # prompt ONLY includes text (point #2: no system/tag/path leakage)
                    gt_block = f"Text: {item.get('source_text', 'N/A')}\n"
                    eval_prompt_for_item = main_eval_prompt + gt_block

                    audio_file = genai.upload_file(path=audio_path)
                    try:
                        resp = call_with_retry(
                            [eval_prompt_for_item, audio_file],
                            gen_config={"temperature": TEMPERATURE, "response_mime_type": "application/json"},
                            max_retries=4,
                        )
                        parsed = robust_json_load(resp.text)
                        parsed_capped = apply_hard_caps(parsed, task_mode=task_mode)

                        with results_lock:
                            results["items"][audio_path] = {
                                "round": round_idx,
                                "status": "ok",
                                "ground_truth": {
                                    "sample_id": item["sample_id"],
                                    "tag": item["tag"],
                                    "system": item["system"],  # stored only for bookkeeping
                                    "audio_path": audio_path,
                                    "source_text": item["source_text"],
                                },
                                "prediction": parsed,
                                "prediction_capped": parsed_capped,
                            }
                            # flush
                            with open(output_json, "w", encoding="utf-8") as f:
                                json.dump(results, f, indent=2, ensure_ascii=False)

                        counters["ok"] += 1

                    finally:
                        try:
                            genai.delete_file(audio_file.name)
                        except Exception as e:
                            write_error_log(audio_path, f"{system_name} ({task_name})", f"delete_file error: {repr(e)}", lock=error_log_lock)

                except Exception as e:
                    counters["error"] += 1
                    write_error_log(audio_path, f"{system_name} ({task_name})", repr(e), lock=error_log_lock)
                    # also store error into results for visibility
                    with results_lock:
                        results["items"][audio_path] = {
                            "round": round_idx,
                            "status": "error",
                            "error": repr(e),
                            "ground_truth": {
                                "sample_id": item.get("sample_id"),
                                "tag": item.get("tag"),
                                "system": item.get("system"),
                                "audio_path": audio_path,
                            },
                        }
                        with open(output_json, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)

            # rounds
            for r in range(N_ROUNDS):
                round_queue = folds[r] if r < len(folds) else []
                rng = random.Random((fold_seed + r * 1000003) & 0xFFFFFFFF)
                rng.shuffle(round_queue)

                # filter already-done
                with results_lock:
                    done = set(results["items"].keys())
                round_queue = [it for it in round_queue if it["audio_path"] not in done]

                if not round_queue:
                    print(f"ℹ️ Round {r+1}/{N_ROUNDS}: no new items.")
                    continue

                print(f"▶ Round {r+1}/{N_ROUNDS}: evaluating {len(round_queue)} samples (~1/3)")

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futures = [ex.submit(process_item, it, r) for it in round_queue]
                    for fu in tqdm(as_completed(futures), total=len(futures), desc=f"{system_name} {task_name} R{r+1}"):
                        # CRITICAL: surface exceptions (otherwise silent)
                        fu.result()

                print(f"   Round {r+1} done. ok={counters['ok']} error={counters['error']} missing={counters['missing']}")

            # summary (system-level mean on capped)
            metric_scores = defaultdict(list)
            for _, rec in results["items"].items():
                if rec.get("status") != "ok":
                    continue
                pred = rec.get("prediction_capped") or {}
                for k, v in pred.items():
                    if k.endswith("_score") and isinstance(v, (int, float)):
                        metric_scores[k].append(float(v))

            results["__summary__"] = {k: {"mean": sum(xs)/len(xs), "n": len(xs)} for k, xs in metric_scores.items()}
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"✅ Done for {system_name} on {task_name}. Results saved to {output_json}")
            print(f"   Summary keys: {list(results['__summary__'].keys())[:5]} ... (total {len(results['__summary__'])})")
            if os.path.exists(missing_log):
                print(f"⚠️ Missing audio list saved to {missing_log}")

    if generated_output_files:
        generate_summary_report(generated_output_files)
    else:
        print("\nNo evaluation performed.")


if __name__ == "__main__":
    main()