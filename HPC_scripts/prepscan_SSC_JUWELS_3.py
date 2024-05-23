from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_SSC_JUWELS_3/"
# learning rate schedule depending on exponential moving average of performance
p["EMA_ALPHA1"]= 0.9
p["EMA_ALPHA2"]= 0.99
p["ETA_FAC"]= 0.1
p["MIN_EPOCH_ETA_FIXED"]= 100
p["TRIAL_MS"]= 1000.0
p["NU_UPPER"]= 10
p["DATA_SET"]= "SSC"
p["N_EPOCH"]= 200
p["EVALUATION"]= "validation_set"
p["DATA_BUFFER_NAME"]= "./data/SSC/mySSC"

# sample obvious parameters
lbd_fac= [ 1.0, 2.0, 5.0, 10.0, 20.0, 50.0 ]
n_batch= [ 32, 64 ]
nu_upper= [ 10, 14 ]
augmentation= [{}, {"random_shift": 5.0}, {"random_shift": 10.0}, {"random_shift": 15.0}, {"random_shift": 20.0}, {"random_shift": 25.0} ]
loss= [ "sum_weigh_exp", "sum_weigh_linear" ]

for i in range(6):
    for j in range(2):
        for k in range(2):
            for l in range(6):
                for m in range(2):
                        id= (((i*2+j)*2+k)*6+l)*2+m
                        print(id)
                        p["LBD_UPPER"]= p0["LBD_UPPER"]*lbd_fac[i]
                        p["LBD_LOWER"]= p0["LBD_LOWER"]*lbd_fac[i]
                        p["N_BATCH"]= n_batch[j]
                        p["NU_UPPER"]= nu_upper[k]
                        p["AUGMENTATION"]= augmentation[l]
                        p["LOSS_TYPE"]= loss[m]
                        p["NAME"]= "JSSC3_scan_"+str(id)
                        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                            json.dump(p, f)
