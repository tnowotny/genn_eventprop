from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["N_BATCH"]= 32
p["N_TRAIN"]= 1000*p["N_BATCH"]

spike_t, spike_ID, rec_vars_n, rec_vars_s= run_yingyang(p)
