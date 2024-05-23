from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_JUWELS_38/"
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
p["AUGMENTATION"]["random_shift"]= 40.0
p["TRIAL_MS"]= 1000.0
p["REWIRE_SILENT"]= True
p["REWIRE_LIFT"]= 0.002

blend = [ [], [0.5, 0.5] ]
n_epoch = [ 200, 100 ]
n_train = [ p0["N_TRAIN"], 2*p0["N_TRAIN"] ]
dt_ms= [ 1, 2, 5, 10 ]
taum= [ 1.0, 2.0, 4.0, 8.0 ]
taus= [ 1.0, 2.0, 4.0, 8.0 ]
train_tau_m = [False, True]
hid_neuron = ["LIF", "hetLIF"]
lbd= [ 2.0, 4.0, 8.0, 16.0 ]

for i in range(2):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                for m in range(2):
                    for n in range(2):
                        for o in range(4):
                            id= (((((i*4+j)*4+k)*4+l)*2+m)*2+n)*4+o
                            print(id)
                            if len(blend[i]) > 0:
                                p["AUGMENTATION"]["blend"] = blend[i]
                            p["N_EPOCH"] = n_epoch[i]
                            p["N_TRAIN"] = n_train[i]
                            p["DT_MS"] = dt_ms[j]
                            p["TAU_MEM"]= p0["TAU_MEM"]*taum[k]
                            p["TAU_MEM_OUTPUT"]= p0["TAU_MEM_OUTPUT"]*taum[k]
                            p["TAU_SYN"]= p0["TAU_SYN"]*taus[l]
                            p["TRAIN_TAUM"] = train_tau_m[m]
                            p["HIDDEN_NEURON_TYPE"] = hid_neuron[n]
                            p["LBD_UPPER"] = p0["LBD_UPPER"]*lbd[o]
                            p["LBD_LOWER"] = p0["LBD_LOWER"]*lbd[o]
                            p["SPK_REC_STEPS"] = int(p["TRIAL_MS"]/p["DT_MS"])
                        
                            p["NAME"] = "J38_scan_"+str(id)
                            with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                json.dump(p, f)
