from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

if len(sys.argv) != 4:
    print("usage: python train_SHD_from_json.py XXX.json OUTDIR NAME")
    exit(1)

with open(sys.argv[1],"r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= sys.argv[2]
p["NAME"]= sys.argv[3]

print(p)
fname= p["NAME"]+".json"
with open(os.path.join(p["OUT_DIR"], fname),"w") as f:
    json.dump(p, f)

mn= SHD_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
