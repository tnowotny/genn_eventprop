from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["N_BATCH"]= 32
p["N_TRAIN"]= 100000*p["N_BATCH"]
p["W_REPORT_INTERVAL"]= 1000
spike_t, spike_ID, rec_vars_n, rec_vars_s= run_yingyang(p)
