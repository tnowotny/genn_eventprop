from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

fname= sys.argv[1]
with open(fname,"r") as f:
    p= json.load(f)

mn= mnist_model(p)
res= mn.cross_validate_SHD(p)

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_summary.json')) as f:
    json.dump(res, f)

