from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

fname= sys.argv[1]
with open(fname,"r") as f:
    p= json.load(f)

mn= mnist_model(p)
res, times= mn.cross_validate_SHD(p)

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_summary.json'),'w') as f:
    json.dump(res, f)

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_times.txt'),'w') as f:
    for t in times:
        f.write("{}\n".format(t))
