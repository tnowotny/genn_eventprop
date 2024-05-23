from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_SSC_JUWELS_5/JSSC5_scan_108.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_SSC_JUWELS_13/"
# learning rate schedule depending on exponential moving average of performance
p["EMA_ALPHA1"]= 0.9
p["EMA_ALPHA2"]= 0.99
p["ETA_FAC"]= 0.1
p["MIN_EPOCH_ETA_FIXED"]= 100
p["TRIAL_MS"]= 1000.0
p["DATA_SET"]= "SSC"
p["N_EPOCH"]= 200
p["EVALUATION"]= "validation_set"
p["DATA_BUFFER_NAME"]= "./data/SSC/mySSC"
p["N_BATCH"]= 32
p["LOSS_TYPE"]= "sum_weigh_exp"
p["AUGMENTATION"]= []
p["ETA"]= 1e-3
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["REWIRE_SILENT"]= True
p["REWIRE_LIFT"]= 0.002
p["BALANCE_TRAIN_CLASSES"]= True
p["COLLECT_CONFUSION"]= True

# sample obvious parameters
num_hidden= [ 128, 256 ]
lbd_fac= [ 0.01, 0.05, 0.1, 0.5 ]
recurrent= [ False, True ]
pdrop= [ 0.0, 0.1 ]
n_hid_layer= [ 1, 2 ]
augment= [ {}, {'random_shift': 5.0}]
rescale_x= [ 1.0, 0.1 ]
for i in range(2):
    for j in range(4):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        for o in range(2):
                            id= (((((i*4+j)*2+k)*2+l)*2+m)*2+n)*2+o
                            print(id)
                            p["NUM_HIDDEN"]= num_hidden[i]
                            p["LBD_UPPER"]= p0["LBD_UPPER"]*lbd_fac[j]
                            p["LBD_LOWER"]= p0["LBD_LOWER"]*lbd_fac[j]
                            p["RECURRENT"]= recurrent[k]
                            p["PDROP_INPUT"]= pdrop[l]
                            p["N_HID_LAYER"]= n_hid_layer[m]
                            p["AUGMENTATION"]= augment[n]
                            p["NAME"]= "JSSC13_scan_"+str(id)
                            p["RESCALE_X"]= rescale_x[o]
                            with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                json.dump(p, f)
