from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_31/J31_scan_769.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_JUWELS_37/"
# learning rate schedule depending on exponential moving average of performance
p["MIN_EPOCH_ETA_FIXED"]= 50
p["EMA_ALPHA2"]= 0.85
p["EMA_ALPHA1"]= 0.8
p["ETA_FAC"]= 0.5
p["N_HID_LAYER"]= 1
p["BALANCE_TRAIN_CLASSES"]= False
p["BALANCE_EVAL_CLASSES"]= False
p["RESCALE_T"]= 1.0
p["RESCALE_X"]= 1.0

# shift augmentation settings
shift= [ 0.0, 10.0, 20.0, 30.0, 40.0, 50.0 ]

# blend augmentation settings
blend= [ [], [0.5, 0.5], [0.8, 0.2], [0.33, 0.33, 0.33] ]
n_epoch= [ 300, 100, 100, 100 ]
n_train= [ p0["N_TRAIN"], 3*p0["N_TRAIN"], 3*p0["N_TRAIN"], 3*p0["N_TRAIN"] ]

# dilation augmentation settings
dilate_min= [ 0.5, 0.8, 0.9, 1.0 ]
dilate_max= [ 2.0, 1.25, 1.1, 1.0 ]

# ID jitter
jitter= [ 0, 5, 10, 20 ]

# train tau_m
train_tau_m= [False, True]

# trial_ms
trial_ms= [ 1000.0, 800.0, 600.0 ]

for i in range(6):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                for m in range(2):
                    for n in range(3):
                        id= ((((i*4+j)*4+k)*4+l)*2+m)*3+n
                        print(id)
                        if shift[i] > 0.0:
                            p["AUGMENTATION"]["random_shift"]= shift[i]

                        if len(blend[j]) > 0:
                            p["AUGMENTATION"]["blend"]= blend[j]
                        p["N_EPOCH"]= n_epoch[j]
                        p["N_TRAIN"]= n_train[j]
                        if dilate_min[k] != 1.0:
                            p["AUGMENTATION"]["random_dilate"]= [dilate_min[k], dilate_max[k]]

                        if jitter[l] != 0:
                            p["AUGMENTATION"]["ID_jitter"]= jitter[l]
                        
                        p["TRAIN_TAUM"]= train_tau_m[m]
                        p["TRIAL_MS"]= trial_ms[n]
                        p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
                        
                        p["NAME"]= "J37_scan_"+str(id)
                        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                            json.dump(p, f)
