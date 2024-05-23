from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_JUWELS_47c/"
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
p["DATA_SET"]= "SHD"
p["N_BATCH"]= 32
p["LOSS_TYPE"]= "sum_weigh_exp"
p["REWIRE_SILENT"]= True
p["REWIRE_LIFT"]= 0.002
p["TAU_OUTPUT_EPOCH_TRIAL"]= []
p["COLLECT_CONFUSION"]= True
p["RECURRENT"]= True
p["INPUT_DELAY"]= 30.0
p["LR_EASE_IN_FACTOR"] = 1.05
p["SPEAKER_LEFT"] = list(range(10))

dt_ms= [ 1, 2, 5, 10, 20 ]
num_hidden= [ 64, 128, 512 ]
delay = [ 0, 10 ]
shift = [ 0.0, 40.0 ]
blend = [ [], [0.5, 0.5] ]
n_epoch = [ 100, 50 ]
n_train = [ p0["N_TRAIN"], 2*p0["N_TRAIN"] ]
train_tau = [False, True]
hid_neuron = ["LIF", "hetLIF"]
lbd_fac= [ 0.1, 0.5, 1.0, 5.0, 10.0 ]
seeds_add = [ 11, 22 ]

for i in range(2):
    for j in range(3):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        for o in range(2):
                            for q in range(5):
                                for r in range(2):
                                    id= (((((((i*3+j)*2+k)*2+l)*2+m)*2+n)*2+o)*5+q)*2+r
                                    print(id)
                                    p["AUGMENTATION"]= {}
                                    p["DT_MS"] = dt_ms[i]
                                    p["NUM_HIDDEN"] = num_hidden[j]
                                    p["N_INPUT_DELAY"] = delay[k]
                                    if delay[k] == 10:
                                        p["INPUT_HIDDEN_MEAN"]= p0["INPUT_HIDDEN_MEAN"]/10.0 # decrease for delay line
                                        p["INPUT_HIDDEN_STD"]= p0["INPUT_HIDDEN_STD"]/10.0
                                    else:
                                        p["INPUT_HIDDEN_MEAN"]= p0["INPUT_HIDDEN_MEAN"]
                                        p["INPUT_HIDDEN_STD"]= p0["INPUT_HIDDEN_STD"]
                                    if shift[l] != 0.0:
                                        p["AUGMENTATION"]["random_shift"] = shift[l]
                                    if len(blend[m]) > 0:
                                        p["AUGMENTATION"]["blend"] = blend[m]
                                    p["N_EPOCH"] = n_epoch[m]
                                    p["N_TRAIN"] = n_train[m]
                                    p["HIDDEN_NEURON_TYPE"] = hid_neuron[n]
                                    p["TRAIN_TAU"] = train_tau[o]
                                    p["LBD_UPPER"] = p0["LBD_UPPER"]*lbd_fac[q]
                                    p["LBD_LOWER"] = p0["LBD_LOWER"]*lbd_fac[q]
                                    p["TRAIN_DATA_SEED"]= p0["TRAIN_DATA_SEED"]+seeds_add[r]
                                    p["TEST_DATA_SEED"]= p0["TEST_DATA_SEED"]+seeds_add[r]
                                    p["MODEL_SEED"]= p0["MODEL_SEED"]+seeds_add[r]
                                    p["SPK_REC_STEPS"] = int(p["TRIAL_MS"]/p["DT_MS"])
                                    p["NAME"] = "J47c_scan_"+str(id)
                                    if not (k == 1 and l == 1 and m == 1):
                                        print(f"k: {k}, l: {l}, m: {m}")
                                        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                            json.dump(p, f)
