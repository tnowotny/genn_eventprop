from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json

with open(sys.argv[1],"r") as f:
    p= json.load(f)

mn= SHD_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
