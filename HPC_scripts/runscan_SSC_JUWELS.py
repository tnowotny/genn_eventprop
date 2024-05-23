from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

fname= sys.argv[1]
with open(fname,"r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["DATA_SET"]= "SSC"
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])

mn= SHD_model(p)
res= mn.train(p)

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_summary.json'),'w') as f:
    json.dump(res, f)

