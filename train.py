from simulator import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

p["DT_MS"]= 1.0
p["ETA"]= 1e-3
p["ETA_DECAY"]= 0.998
p["ADAM_BETA1"]= 0.95
p["ADAM_BETA2"]= 0.9995    
p["ALPHA"]= 5e-2 #3e-3
p["N_BATCH"]= 512
p["N_TRAIN"]= 10*p["N_BATCH"]
p["N_TEST"]= 10*p["N_BATCH"]
p["W_REPORT_INTERVAL"]= 3000
p["W_EPOCH_INTERVAL"] = 100
p["N_EPOCH"]= 400
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["TRAINING_PLOT"]= False
p["TRAINING_PLOT_INTERVAL"]= 1
p["INPUT_HIDDEN_MEAN"]= 1.5
p["INPUT_HIDDEN_STD"]= 0.78
p["LOAD_LAST"]= False


x= [-3.     , -2.69897, -1.30103, -3.30103, -1.30103]

log10= np.log(10)
p["ETA"]= np.exp(x[0]*log10)
p["ETA_DECAY"]= 1-np.exp(x[1]*log10)
p["ADAM_BETA1"]= 1-np.exp(x[2]*log10)
p["ADAM_BETA2"]= 1-np.exp(x[3]*log10)
p["ALPHA"]= np.exp(x[4]*log10)
p["LOAD_LAST"]= False

yy= yingyang(p)
yy.train(p)

p["LOAD_LAST"]= True
p["TRAINING_PLOT"]= True
p["FANCY_PLOTS"]= True
yy.test(p)

