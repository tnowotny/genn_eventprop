from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["DT_MS"]= 0.1
p["ETA"]= 1e-3
p["ETA_DECAY"]= 0.998
p["ADAM_BETA1"]= 0.98
p["ADAM_BETA2"]= 0.9998    
p["ALPHA"]= 5e-2 #3e-3
p["N_BATCH"]= 512
p["N_TRAIN"]= 10*p["N_BATCH"]
p["N_TEST"]= 500*p["N_BATCH"]
p["W_REPORT_INTERVAL"]= 3000
p["W_EPOCH_INTERVAL"] = 100
p["N_EPOCH"]= 500
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["TRAINING_PLOT"]= False
p["TRAINING_PLOT_INTERVAL"]= 1
p["INPUT_HIDDEN_MEAN"]= 1.5
p["INPUT_HIDDEN_STD"]= 0.78
p["LOAD_LAST"]= False
spike_t, spike_ID, rec_vars_n, rec_vars_s= run_yingyang(p)
