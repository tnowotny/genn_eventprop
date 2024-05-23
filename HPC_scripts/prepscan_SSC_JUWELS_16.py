from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_SSC_JUWELS_16/"
# learning rate schedule depending on exponential moving average of performance
p["MIN_EPOCH_ETA_FIXED"]= 50
p["EMA_ALPHA2"]= 0.85
p["EMA_ALPHA1"]= 0.8
p["ETA_FAC"]= 0.5
p["TRIAL_MS"]= 1000.0
p["NU_UPPER"]= 14
p["N_HID_LAYER"]= 1
p["BALANCE_TRAIN_CLASSES"]= False
p["BALANCE_EVAL_CLASSES"]= False
p["RESCALE_T"]= 1.0
p["RESCALE_X"]= 1.0
p["DATA_SET"]= "SSC"
p["EVALUATION"]= "validation_set"
p["DATA_BUFFER_NAME"]= "./data/SSC/mySSC"
p["N_BATCH"]= 32
p["LOSS_TYPE"]= "sum_weigh_exp"
p["REWIRE_SILENT"]= True
p["REWIRE_LIFT"]= 0.002
p["TRAIN_TAU"]= True
p["TAU_OUTPUT_EPOCH_TRIAL"]= [ [0,0], [99,0], [199,0] ]
p["COLLECT_CONFUSION"]= True
p["RECURRENT"]= True
p["AUGMENTATION"]= {
    "random_shift": 40.0
}

p["N_INPUT_DELAY"]= 10
p["INPUT_HIDDEN_MEAN"]= p0["INPUT_HIDDEN_MEAN"]/10.0 # reduce to keep initial activity under control
p["INPUT_HIDDEN_STD"]= p0["INPUT_HIDDEN_STD"]/10.0
p["INPUT_DELAY"]= 30.0

blend = [ [], [0.5, 0.5] ]
n_epoch = [ 200, 100 ]
n_train = [ p0["N_TRAIN"], 150932 ] # for non-blend, this number is meaningless, for blend I put explicitly 2x number of samples in SSC train
dt_ms= [ 1, 2, 5 ]
taum= [ 1.0, 2.0 ]
taus= [ 2.0, 4.0 ]
train_tau = [False, True]
hid_neuron = ["LIF", "hetLIF"]
num_hidden= [ 256, 1024 ]
lbd_fac= [ 0.1, 0.5, 1.0, 5.0, 10.0 ]

for i in range(2):
    for j in range(3):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        for o in range(2):
                            for q in range(5):
                                id= ((((((i*3+j)*2+k)*2+l)*2+m)*2+n)*2+o)*5+q
                                print(id)
                                if len(blend[i]) > 0:
                                    p["AUGMENTATION"]["blend"] = blend[i]
                                p["N_EPOCH"] = n_epoch[i]
                                p["N_TRAIN"] = n_train[i]
                                p["DT_MS"] = dt_ms[j]
                                p["TAU_MEM"]= p0["TAU_MEM"]*taum[k]
                                p["TAU_MEM_OUTPUT"]= p0["TAU_MEM_OUTPUT"]*taum[k]
                                p["TAU_SYN"]= p0["TAU_SYN"]*taus[l]
                                p["TRAIN_TAU"] = train_tau[m]
                                p["HIDDEN_NEURON_TYPE"] = hid_neuron[n]
                                p["NUM_HIDDEN"] = num_hidden[o]
                                p["LBD_UPPER"] = p0["LBD_UPPER"]*lbd_fac[q]
                                p["LBD_LOWER"] = p0["LBD_LOWER"]*lbd_fac[q]
                                p["SPK_REC_STEPS"] = int(p["TRIAL_MS"]/p["DT_MS"])
                                p["NAME"]= "JSSC16_scan_"+str(id)
                                with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                    json.dump(p, f)
