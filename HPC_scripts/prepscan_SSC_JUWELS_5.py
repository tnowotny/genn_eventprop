from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_SSC_JUWELS_5/"
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
# sample obvious parameters
lbd_fac= [ 1.0, 2.0, 5.0, 10.0, 20.0, 50.0 ]
recurrent= [ False, True ]
n_hid_layer= [ 1, 2, 3 ]
augmentation= [{"random_shift": 10.0}, {"random_shift": 15.0}, {"random_shift": 20.0}]
hiddenfwd_mean= [ 0.06, 0.09 ]
hiddenfwd_std= [ 0.02, 0.03 ]
num_hidden= [ 128, 256, 512 ]

for i in range(6):
    for j in range(2):
        for k in range(3):
            for l in range(3):
                for m in range(2):
                    for n in range(2):
                        for o in range(3):
                            id= (((((i*2+j)*3+k)*3+l)*2+m)*2+n)*3+o
                            print(id)
                            p["LBD_UPPER"]= p0["LBD_UPPER"]*lbd_fac[i]
                            p["LBD_LOWER"]= p0["LBD_LOWER"]*lbd_fac[i]
                            p["RECURRENT"]= recurrent[j]
                            p["N_HID_LAYER"]= n_hid_layer[k]
                            p["AUGMENTATION"]= augmentation[l]
                            p["HIDDEN_HIDDENFWD_MEAN"]= hiddenfwd_mean[m]
                            p["HIDDEN_HIDDENFWD_STD"]= hiddenfwd_std[n]
                            p["NUM_HIDDEN"]= num_hidden[o]
                            p["NAME"]= "JSSC5_scan_"+str(id)
                            with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                json.dump(p, f)
