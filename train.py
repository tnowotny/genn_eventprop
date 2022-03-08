from simulator import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

p["DT_MS"]= 0.1
p["ETA"]= 5e-3
p["ETA_DECAY"]= 0.95
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999    
p["ALPHA"]= 1e-3
p["N_BATCH"]= 512
p["N_TRAIN"]= 10*p["N_BATCH"]
p["N_TEST"]= 2*p["N_BATCH"]
p["W_REPORT_INTERVAL"]= 3000
p["W_EPOCH_INTERVAL"] = 100
p["N_EPOCH"]= 100
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["TRAINING_PLOT"]= False
p["TRAINING_PLOT_INTERVAL"]= 1
p["INPUT_HIDDEN_MEAN"]= 1.5
p["INPUT_HIDDEN_STD"]= 0.78
p["LOAD_LAST"]= False


x= [-3.     , -2.69897, -1.30103, -3.30103, -1.30103]
x= [-2.5     , -2.69897, -1, -2, -1]

log10= np.log(10)
p["ETA"]= np.exp(x[0]*log10)
p["ETA_DECAY"]= 1-np.exp(x[1]*log10)
p["ADAM_BETA1"]= 1-np.exp(x[2]*log10)
p["ADAM_BETA2"]= 1-np.exp(x[3]*log10)
p["ALPHA"]= np.exp(x[4]*log10)

yy= yingyang(p)
yy.train(p)

p["LOAD_LAST"]= True
p["TRAINING_PLOT"]= True
p["FANCY_PLOTS"]= True
yy.test(p)

