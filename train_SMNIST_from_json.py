from simulator_SMNIST import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

if len(sys.argv) != 3:
    print("usage: python train_SMNIST_from_json.py XXX.json NAME")
    exit(1)

with open(sys.argv[1],"r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["NAME"]= sys.argv[2]

mn= SMNIST_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
