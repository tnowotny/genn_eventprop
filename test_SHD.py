from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np

from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np
import json

p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= 366
p["TEST_DATA_SEED"]= 149
p["NAME"]= "test31"
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 1500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["DEBUG"]= True
p["DEBUG_HIDDEN_N"]= True
p["N_EPOCH"]= 1
p["N_BATCH"]= 256
p["N_TRAIN"]= 7900 #20*p["N_BATCH"] #7756 
p["N_VALIDATE"]= 256 # 256 # p["N_BATCH"] 
p["ETA"]= 1e-3 #1e-2 #5e-3
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.02
p["INPUT_HIDDEN_STD"]= 0.03
p["HIDDEN_OUTPUT_MEAN"]= 0.1
p["HIDDEN_OUTPUT_STD"]= 0.3
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 0.000000000001 # keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 0.005
p["NU_UPPER"]= 17*p["N_BATCH"]
p["NU_LOWER"]= 0.1*p["N_BATCH"]
p["RHO_UPPER"]= 5000.0
p["GLB_UPPER"]= 0.0
p["ETA_DECAY"]= 1.0      
p["ETA_FIDDELING"]= False
p["WRITE_TO_DISK"]= False
p["LOAD_LAST"]= True
p["TIMING"]= False
p["LOSS_TYPE"]= "sum"
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])

if p["DEBUG"]:
    p["REC_SPIKES"]= ["input", "hidden"]

mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.test(p)

if p["DEBUG"]:
    plt.figure()
    plt.scatter(spike_t["input"],700-spike_ID["input"],s=0.5)
    plt.figure()
    plt.scatter(spike_t["hidden"],spike_ID["hidden"],s=0.5)
    tmax= np.max(spike_t["hidden"])
    t= np.arange(0,tmax,p["TRIAL_MS"])
    v= np.vstack([ np.zeros((1,len(t))), np.ones((1,len(t)))])
    t= np.reshape(t,(1,len(t)))
    t= np.vstack([ t, t ])
    plt.plot(t,v, lw=0.1)
    plt.figure()
    h, x= np.histogram(spike_ID["hidden"],p["NUM_HIDDEN"])
    h= np.sort(h)
    plt.bar(np.arange(len(h)),np.log(h+2))
mn.hid_to_out.pull_var_from_device("w")
plt.figure()
plt.hist(mn.hid_to_out.vars["w"].view[:],100)
plt.show()
