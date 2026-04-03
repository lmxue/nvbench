import sys

fr = open(sys.argv[1], mode="r", encoding="utf-8")

OVRL_raw,SIG_raw,BAK_raw,OVRL,SIG,BAK = [],[],[],[],[],[]

for i in fr:
    a = i.split(',')
    if len(a[0]) == 0:
        continue
    # BAK,BAK_raw,OVRL,OVRL_raw,SIG,SIG_raw
    # OVRL_raw.append(float(a[4]))
    # SIG_raw.append(float(a[6]))
    # BAK_raw.append(float(a[2]))
    # OVRL.append(float(a[3]))
    # SIG.append(float(a[5]))
    # BAK.append(float(a[1]))
    OVRL_raw.append(float(a[5]))
    SIG_raw.append(float(a[6]))
    BAK_raw.append(float(a[7]))
    OVRL.append(float(a[8]))
    SIG.append(float(a[9]))
    BAK.append(float(a[10]))

print("num: ", len(OVRL_raw))
print("OVRL_raw: {:.4f}".format(sum(OVRL_raw)/len(OVRL_raw)))
print("SIG_raw: {:.4f}".format(sum(SIG_raw)/len(SIG_raw)))
print("BAK_raw: {:.4f}".format(sum(BAK_raw)/len(BAK_raw)))
print("OVRL: {:.4f}".format(sum(OVRL)/len(OVRL)))
print("SIG: {:.4f}".format(sum(SIG)/len(SIG)))
print("BAK: {:.4f}".format(sum(BAK)/len(BAK)))
    
fr.close()