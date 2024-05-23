from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_SSC_JUWELS_1/"
# learning rate schedule depending on exponential moving average of performance
p["EMA_ALPHA1"]= 0.9
p["EMA_ALPHA2"]= 0.99
p["ETA_FAC"]= 0.5
p["MIN_EPOCH_ETA_FIXED"]= 300
p["TRIAL_MS"]= 1000.0
p["DATA_SET"]= "SSC"
p["N_EPOCH"]= 100
p["EVALUATION"]= "validation_set"
p["DATA_BUFFER_NAME"]= "./data/SSC/mySSC"
# sample temporal scaling
lbd= [ 0.2, 0.5, 1.0, 2.0 ]

for i in range(4):
    id= i
    print(id)
    p["LBD_UPPER"]= p0["LBD_UPPER"]*lbd[i]
    p["LBD_LOWER"]= p0["LBD_LOWER"]*lbd[i]
    p["NAME"]= "JSSC1_scan_"+str(id)
    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
        json.dump(p, f)
