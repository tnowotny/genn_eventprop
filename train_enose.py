from simulator_enose import *
import matplotlib.pyplot as plt
import numpy as np
import json

p["TRIAL_MS"]= 100
p["DATASET"]= "enose"
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["NAME"]= "enose0"
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 1500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["DEBUG"]= False
p["DEBUG_HIDDEN_N"]= False
p["LOAD_LAST"]= False
p["N_EPOCH"]= 500
p["N_BATCH"]= 32
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 940 # 1004 #20*p["N_BATCH"] #7756 
p["N_VALIDATE"]= 64 # 256 # p["N_BATCH"] 
p["ETA"]= 5e-3 #5e-3
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.4
p["INPUT_HIDDEN_STD"]= 0.2
p["HIDDEN_OUTPUT_MEAN"]= 0.0
p["HIDDEN_OUTPUT_STD"]= 0.3
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["REG_TYPE"]= "none"
p["LBD_UPPER"]= 8e-16 # keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 1e-5
p["NU_UPPER"]= 15 #*p["N_BATCH"]
p["NU_LOWER"]= 5
p["RHO_UPPER"]= 10000.0
p["GLB_UPPER"]= 1e-8
p["ETA_DECAY"]= 1.0      
p["ETA_FIDDELING"]= False
p["ETA_REDUCE"]= 0.5
p["ETA_REDUCE_PERIOD"]= 50
p["TIMING"]= False
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["LOSS_TYPE"]= "sum"
p["EVALUATION"]= "random"

p["RECURRENT"]= False
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02

if p["DEBUG"]:
    p["REC_SPIKES"]= ["input", "hidden"]
    p["REC_NEURONS"]= [("output", "V")] #, ("output", "lambda_V"), ("output", "lambda_I")]
    #p["REC_SYNAPSES"]= [("hid_to_out", "w")]

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as file:
    json.dump(p, file)
    
mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
if p["DEBUG"]:
    plt.figure()
    plt.scatter(spike_t["input"],8-spike_ID["input"],s=0.5)
    plt.figure()
    plt.scatter(spike_t["hidden"],spike_ID["hidden"],s=0.5)
    #tmax= np.max(spike_t["hidden"])
    #t= np.arange(0,tmax,p["TRIAL_MS"])
    #v= np.vstack([ np.zeros(1,lent(t)), np.ones(1,len(t))])
    #t= np.reshape(t,(1,len(t)))
    #t= np.vstack([ t, t ])
    #plt.plot(t,v, lw=0.1)
    plt.figure()
    plt.plot(rec_vars_n["Voutput"])
    #plt.figure()
    #plt.plot(rec_vars_n["lambda_Voutput"])
    #plt.figure()
    #plt.plot(rec_vars_n["lambda_Ioutput"])
    #print(mn.output.extra_global_params["label"].view[0:20*p["N_BATCH"]:p["N_BATCH"]])
    #plt.figure()
    #h, x= np.histogram(spike_ID["hidden"],p["NUM_HIDDEN"])
    #h= np.sort(h)
    #plt.bar(np.arange(len(h)),np.log(h+2))
#mn.hid_to_out.pull_var_from_device("w")
#plt.figure()
#plt.hist(mn.hid_to_out.vars["w"].view[:],100)
plt.show()
