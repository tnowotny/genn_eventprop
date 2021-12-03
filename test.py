from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["N_BATCH"]= 32
p["TRAIN"]= False
p["N_EPOCH"]= 1
spike_t, spike_ID, rec_vars_n, rec_vars_s= run_yingyang(p)
