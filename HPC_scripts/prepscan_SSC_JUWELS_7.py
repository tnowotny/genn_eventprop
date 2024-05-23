from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_SSC_JUWELS_7/"
# learning rate schedule depending on exponential moving average of performance
p["EMA_ALPHA1"]= 0.9
p["EMA_ALPHA2"]= 0.99
p["ETA_FAC"]= 0.1
p["MIN_EPOCH_ETA_FIXED"]= 100
p["TRIAL_MS"]= 1000.0
p["DATA_SET"]= "SSC"
p["N_EPOCH"]= 400
p["EVALUATION"]= "validation_set"
p["DATA_BUFFER_NAME"]= "./data/SSC/mySSC"
p["N_BATCH"]= 32
p["NUM_HIDDEN"]= 256
p["LOSS_TYPE"]= "sum_weigh_exp"
p["AUGMENTATION"]= []
# sample obvious parameters
eta_fac= [ 0.1, 0.2, 0.5 ] 
lbd_fac= [ 1.0, 5.0, 20.0, 100.0 ]
nu_upper= [ 10.0, 15.0, 20.0 ]
recurrent= [ False, True ]
n_hid_layer= [ 1, 2, 3 ]
hiddenfwd_mean= [ 0.06, 0.09 ]
hiddenfwd_std= [ 0.02, 0.03 ]

for i in range(3):
    for j in range(4):
        for k in range(3):
            for l in range(2):
                for m in range(3):
                    for n in range(2):
                        for o in range(2):
                            id= (((((i*4+j)*3+k)*2+l)*3+m)*2+n)*2+o
                            print(id)
                            p["ETA"]= p0["ETA"]*eta_fac[i]
                            p["LBD_UPPER"]= p0["LBD_UPPER"]*lbd_fac[j]
                            p["LBD_LOWER"]= p0["LBD_LOWER"]*lbd_fac[j]
                            p["NU_UPPER"]= nu_upper[k]
                            p["RECURRENT"]= recurrent[l]
                            p["N_HID_LAYER"]= n_hid_layer[m]
                            p["HIDDEN_HIDDENFWD_MEAN"]= hiddenfwd_mean[n]
                            p["HIDDEN_HIDDENFWD_STD"]= hiddenfwd_std[o]
                            p["NAME"]= "JSSC7_scan_"+str(id)
                            with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                json.dump(p, f)
