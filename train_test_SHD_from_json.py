from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

if len(sys.argv) != 4:
    print("usage: python train_SHD_from_json.py XXX.json <outdir> <name>")
    exit(1)

with open(sys.argv[1],"r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= sys.argv[2]
p["NAME"]= sys.argv[3]

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as file:
    json.dump(p, file)

mn= SHD_model(p)

p["N_TRAIN"]= 8156
p["N_VALIDATE"]= 0
p["CUDA_VISIBLE_DEVICES"]= False

for i in range(1):
    mn= SHD_model(p)
    spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train_test(p)
    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_traintest.txt'),'a') as f:
        f.write("{} {}\n".format(correct,correct_eval))
    p["TRAIN_DATA_SEED"]+= 31
