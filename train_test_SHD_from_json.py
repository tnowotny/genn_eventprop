from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

if len(sys.argv) != 3:
    print("usage: python train_SHD_from_json.py XXX.json NAME")
    exit(1)

with open(sys.argv[1],"r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["NAME"]= sys.argv[2]

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as file:
    json.dump(p, file)

mn= SHD_model(p)

for i in range(10):
    p["LOAD_LAST"]= False
    spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
    print("training correct: {}".format(correct))
    print("training correct_eval: {}".format(correct_eval))
    tc= correct
    p["TRAIN_DATA_SEED"]+= 31
    p["LOAD_LAST"]= True
    spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.test(p)
    print("test correct: {}".format(correct))
    print("test correct_eval: {}".format(correct_eval))
    p["TEST_DATA_SEED"]+= 31
    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_allresult.txt'),'a') as f:
        f.write("{} {}\n".format(tc,correct_eval))
