from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_SSC_JUWELS_2/"
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
lbd_fac= [ 0.1, 0.3, 1.0, 3.0 ]
eta_fac= [ 0.1, 0.3, 1.0, 3.0 ]
n_batch= [ 32, 64, 128, 256 ]
augmentation= [ {}, {"random_shift": 20.0}, {"random_shift": 40.0} ]
train_taum= [ False, True ]
loss= [ "sum_weigh_exp", "sum_weigh_linear" ]

for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(3):
                for m in range(2):
                    for n in range(2):
                        id= ((((i*4+j)*4+k)*3+l)*2+m)*2+n
                        print(id)
                        p["LBD_UPPER"]= p0["LBD_UPPER"]*lbd_fac[i]
                        p["LBD_LOWER"]= p0["LBD_LOWER"]*lbd_fac[i]
                        p["ETA"]= p0["ETA"]*eta_fac[j]
                        p["N_BATCH"]= n_batch[k]
                        p["AUGMENTATION"]= augmentation[l]
                        p["TRAIN_TAUM"]= train_taum[m]
                        p["LOSS_TYPE"]= loss[n]
                        p["NAME"]= "JSSC2_scan_"+str(id)
                        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                            json.dump(p, f)
