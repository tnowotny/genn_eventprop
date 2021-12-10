from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["DT_MS"]= 0.1
p["ETA"]= 1e-3
p["ETA_DECAY"]= 0.99      
p["ALPHA"]= 3e-4
p["N_BATCH"]= 512
p["N_TRAIN"]= 100*p["N_BATCH"]
p["N_TEST"]= 100*p["N_BATCH"]
p["W_REPORT_INTERVAL"]= 3000
p["W_EPOCH_INTERVAL"] = 100
p["N_EPOCH"]= 1000
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["TRAINING_PLOT"]= False
p["TRAINING_PLOT_INTERVAL"]= 1
spike_t, spike_ID, rec_vars_n, rec_vars_s= run_yingyang(p)
