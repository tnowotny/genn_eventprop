from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["DT_MS"]= 0.1
p["N_BATCH"]= 32
p["N_TEST"]= 20*p["N_BATCH"]
p["TRAIN"]= False
p["N_EPOCH"]= 1
p["REC_SPIKES"]=["input"]
p["FANCY_PLOTS"]= True
spike_t, spike_ID, rec_vars_n, rec_vars_s= run_yingyang(p)


