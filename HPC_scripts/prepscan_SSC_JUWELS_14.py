from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_SSC_JUWELS_14/"
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
p["N_BATCH"]= 64
p["LOSS_TYPE"]= "sum_weigh_exp"
p["REWIRE_SILENT"]= True
p["REWIRE_LIFT"]= 0.003
p["HIDDEN_NEURON_TYPE"]= "hetLIF"

# sample obvious parameters
lbd_fac= [ 0.1, 0.5, 1.0, 5.0, 10.0 ]
eta_fac= [ 0.1, 0.5, 1.0, 5.0, 10.0 ]
recurrent= [ False, True ]
augmentation= [{}, {"random_shift": 5.0}, {"random_shift": 10.0}, {"random_shift": 20.0}]
num_hidden= [ 128, 256, 512, 1024 ]

for i in range(5):
    for j in range(5):
        for k in range(2):
            for l in range(4):
                for m in range(4):
                    id= (((i*5+j)*2+k)*4+l)*4+m
                    print(id)
                    p["LBD_UPPER"]= p0["LBD_UPPER"]*lbd_fac[i]
                    p["LBD_LOWER"]= p0["LBD_LOWER"]*lbd_fac[i]
                    p["ETA"]= p0["ETA"]*eta_fac[j]
                    p["RECURRENT"]= recurrent[k]
                    p["AUGMENTATION"]= augmentation[l]
                    p["NUM_HIDDEN"]= num_hidden[m]
                    p["NAME"]= "JSSC14_scan_"+str(id)
                    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                        json.dump(p, f)
