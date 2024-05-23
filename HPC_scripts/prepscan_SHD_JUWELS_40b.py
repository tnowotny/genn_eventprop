from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_39b/J39b_scan_482.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_JUWELS_40b/"
# learning rate schedule depending on exponential moving average of performance
p["TAU_OUTPUT_EPOCH_TRIAL"]= []
p["AUGMENTATION"]= {}

p["EMA_WHICH"] = "training"
p["CHECKPOINT_BEST"] = "training"
p["TRAIN_TAU"] = True
p["TRAIN_TAU_OUTPUT"] = True
p["LR_EASE_IN_FACTOR"] = 1.01


dt_ms= [ 1, 2, 5, 10, 20 ]
num_hidden= [ 64, 128, 256, 512, 1024 ]
delay= [ 0, 10 ]
hid_neuron = ["LIF", "hetLIF"]
blend = [ [], [0.5, 0.5] ]
n_epoch = [ 200, 100 ]
n_train = [ p0["N_TRAIN"]/2, p0["N_TRAIN"] ] # for non-blend, this number is meaningless, for blend I put explicitly 2x number of samples in SHD train
shift = [ 0, 40.0 ]
seeds_add = [ 11, 22, 33, 44 ]
for i in range(5):
    for j in range(5):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        for o in range(4):
                            id= (((((i*5+j)*2+k)*2+l)*2+m)*2+n)*4+o
                            print(id)
                            p["AUGMENTATION"]= {}
                            p["DT_MS"] = dt_ms[i]
                            p["NUM_HIDDEN"] = num_hidden[j]
                            p["N_INPUT_DELAY"] = delay[k]
                            if delay[k] == 0:
                                p["INPUT_HIDDEN_MEAN"]= p0["INPUT_HIDDEN_MEAN"]*10.0 # increase as opposed to delay line
                                p["INPUT_HIDDEN_STD"]= p0["INPUT_HIDDEN_STD"]*10.0
                            p["HIDDEN_NEURON_TYPE"] = hid_neuron[l]
                            if len(blend[m]) > 0:
                                p["AUGMENTATION"]["blend"] = blend[m]
                            p["N_EPOCH"] = n_epoch[m]
                            p["N_TRAIN"] = n_train[m]
                            if shift[n] > 0:
                                p["AUGMENTATION"]["random_shift"] = shift[n]
                            p["TRAIN_DATA_SEED"]= p0["TRAIN_DATA_SEED"]+seeds_add[o]
                            p["TEST_DATA_SEED"]= p0["TEST_DATA_SEED"]+seeds_add[o]
                            p["MODEL_SEED"]= p0["MODEL_SEED"]+seeds_add[o]
                            p["SPK_REC_STEPS"] = int(p["TRIAL_MS"]/p["DT_MS"])
                            p["NAME"]= "J40b_scan_"+str(id)
                            with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                json.dump(p, f)
