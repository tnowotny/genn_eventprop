from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_JUWELS_33/"
# learning rate schedule depending on exponential moving average of performance
p["EMA_ALPHA1"]= 0.9
p["EMA_ALPHA2"]= 0.99
p["ETA_FAC"]= 0.5
p["MIN_EPOCH_ETA_FIXED"]= 300
p["TRIAL_MS"]= 1000.0

# sample temporal scaling
dt_ms= [ 1, 2, 5, 10 ]
tau_m= [ 3.0, 4.0, 5.0, 6.0 ]
tau_syn= [ 1.0, 2.0, 3.0, 4.0 ]
lbd= [ 1.0, 2.0, 3.0, 4.0 ]
min_epoch= [ 50, 300 ]
loss= ["sum_weigh_exp", "sum_weigh_linear"]


for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                for m in range(2):
                    for n in range(2):
                        id= ((((i*4+j)*4+k)*4+l)*2+m)*2+n
                        print(id)
                        p["DT_MS"]= dt_ms[i]
                        p["TAU_MEM"]= p0["TAU_MEM"]*tau_m[j]
                        p["TAU_MEM_OUTPUT"]= p0["TAU_MEM_OUTPUT"]*tau_m[j]
                        p["TAU_SYN"]= p0["TAU_SYN"]*tau_syn[k]
                        p["LBD_UPPER"]= p0["LBD_UPPER"]*lbd[l]
                        p["LBD_LOWER"]= p0["LBD_LOWER"]*lbd[l]
                        p["LOSS_TYPE"]= loss[m]
                        p["MIN_EPOCH_ETA_FIXED"]= min_epoch[n]
                        p["NAME"]= "J33_scan_"+str(id)
                        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                            json.dump(p, f)
