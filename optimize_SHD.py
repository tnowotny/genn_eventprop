from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= None
p["TEST_DATA_SEED"]= None
p["NAME"]= "test18"
p["NUM_HIDDEN"]= 512
p["N_MAX_SPIKE"]= 500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.0
p["ADAM_BETA1"]= 0.999
p["ADAM_BETA2"]= 0.99999   
p["DEBUG"]= False
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 20
p["N_BATCH"]= 128
p["N_TRAIN"]= 7808 #20*p["N_BATCH"] #7756 
p["N_VALIDATE"]= 256 #p["N_BATCH"] 
p["ETA"]= 1e-2
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.01
p["INPUT_HIDDEN_STD"]= 0.018
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["REGULARISATION"]= True
p["LBD_UPPER"]= 0.00005
p["LBD_LOWER"]= 0.001
p["NU_UPPER"]= 2000
p["NU_LOWER"]= 2


def costfun(x, shd, p):
    print("x = {}".format(x))
    log10= np.log(10)
    p["ETA"]= np.exp(x[0]*log10)
    p["LDB_UPPER"]= np.exp(x[1]*log10)
    p["LBD_LOWER"]= np.exp(x[2]*log10)
    p["NU_UPPER"]= x[3]*1000
    p["NU_LOWER"]= x[4]
    p["LOAD_LAST"]= False
    spike_t, spike_ID, rec_vars_n, rec_vars_s, correct, correct_eval= shd.train(p)
    return 1-(correct+correct_eval)/2


log10= np.log(10)
x= [ np.log(p["ETA"])/log10,
     np.log(p["LBD_UPPER"])/log10, 
     np.log(p["LBD_LOWER"])/log10,
     p["NU_UPPER"]/1000.0,
     p["NU_LOWER"]
]

bound= [ (-4,-1), (-7,-3), (-6, -2), (0.1, 5), (0, 10) ]

shd= mnist_model(p)
method = 'nelder-mead'
options = {'disp': True, 'maxiter': 50000, 'maxfev': 10, 'xatol': 1e-8, 'fatol': 1e-8}
result = minimize(costfun, x, method=method, options=options,
                  args=(shd, p), bounds= bound)

print(result)





