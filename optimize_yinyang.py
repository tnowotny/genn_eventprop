from simulator import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

p["DT_MS"]= 0.1
p["ETA"]= 1e-3
p["ETA_DECAY"]= 0.998
p["ADAM_BETA1"]= 0.95
p["ADAM_BETA2"]= 0.9995    
p["ALPHA"]= 5e-2 #3e-3
p["N_BATCH"]= 512
p["N_TRAIN"]= 10*p["N_BATCH"]
p["N_TEST"]= 2*p["N_BATCH"]
p["W_REPORT_INTERVAL"]= 3000
p["W_EPOCH_INTERVAL"] = 100
p["N_EPOCH"]= 200
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["TRAINING_PLOT"]= False
p["TRAINING_PLOT_INTERVAL"]= 1
p["INPUT_HIDDEN_MEAN"]= 1.5
p["INPUT_HIDDEN_STD"]= 0.78
p["LOAD_LAST"]= False

yy= yingyang(p)


def costfun(x, yy, p):
    print("x = {}".format(x))
    log10= np.log(10)
    p["ETA"]= np.exp(x[0]*log10)
    p["ETA_DECAY"]= 1-np.exp(x[1]*log10)
    p["ADAM_BETA1"]= 1-np.exp(x[2]*log10)
    p["ADAM_BETA2"]= 1-np.exp(x[3]*log10)
    p["ALPHA"]= np.exp(x[4]*log10)
    p["N_TRAIN"]= int(x[5]*10)*p["N_BATCH"]
    p["N_EPOCH"]= int(x[6]*100)
    p["LOAD_LAST"]= False
    yy.train(p)
    p["LOAD_LAST"]= True
    spike_t, spike_ID, rec_vars_n, rec_vars_s, correct= yy.test(p)
    p["LOAD_LAST"]= False
    yy.train(p)
    p["LOAD_LAST"]= True
    spike_t, spike_ID, rec_vars_n, rec_vars_s, correct2= yy.test(p)
    return (2-correct-correct2)/2

log10= np.log(10)
x= [ np.log(p["ETA"])/log10,
     np.log(1-p["ETA_DECAY"])/log10, 
     np.log(1-p["ADAM_BETA1"])/log10,
     np.log(1-p["ADAM_BETA2"])/log10,
     np.log(p["ALPHA"])/log10,
     1,
     1,
]

bound= [ (-5,-2), (-5,-1), (-3, -0.5), (-6, -2), (-4, -0.5), (0.1, 2), (0.1, 4) ]
     
method = 'nelder-mead'
options = {'disp': True, 'maxiter': 50000, 'maxfev': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
result = minimize(costfun, x, method=method, options=options,
                  args=(yy, p), bounds= bound)

print(result)
