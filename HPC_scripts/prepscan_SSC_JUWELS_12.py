from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_SSC_JUWELS_5/JSSC5_scan_108.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_SSC_JUWELS_12/"
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
p["BALANCE_TRAIN_CLASSES"]= True
p["COLLECT_CONFUSION"]= True
p["RESCALE_X"]= 0.1
p["PDROP_INPUT"]= 0.1

# sample obvious parameters
num_hidden= [ 128, 256 ]
lbd_fac= [ 0.01, 0.05, 0.1, 0.5 ]
recurrent= [ False, True ]
n_input_delay= [ 10, 20 ]
input_delay= [ 10.0, 20.0, 30.0, 40.0 ]
augment= [ {}, {'random_shift': 1.0}]
for i in range(2):
    for j in range(4):
        for k in range(2):
            for l in range(2):
                for m in range(4):
                    for n in range(2):
                        id= ((((i*4+j)*2+k)*2+l)*4+m)*2+n
                        print(id)
                        p["NUM_HIDDEN"]= num_hidden[i]
                        p["LBD_UPPER"]= p0["LBD_UPPER"]*lbd_fac[j]
                        p["LBD_LOWER"]= p0["LBD_LOWER"]*lbd_fac[j]
                        p["RECURRENT"]= recurrent[k]
                        p["N_INPUT_DELAY"]= n_input_delay[l]
                        p["INPUT_HIDDEN_MEAN"]= 0.02/n_input_delay[l] # reduce to keep initial activity under control
                        p["INPUT_HIDDEN_STD"]= 0.2/n_input_delay[l] 
                        p["REWIRE_LIFT"]= 0.02/n_input_delay[l] 
                        p["INPUT_DELAY"]= input_delay[m]
                        p["AUGMENTATION"]= augment[n]
                        p["NAME"]= "JSSC12_scan_"+str(id)
                        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                            json.dump(p, f)
