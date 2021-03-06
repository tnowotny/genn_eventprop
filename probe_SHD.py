from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np

p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["NAME"]= "test20"
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.0
p["ADAM_BETA1"]= 0.999
p["ADAM_BETA2"]= 0.99999   
p["DEBUG"]= True
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 1
p["N_BATCH"]= 128
p["N_TRAIN"]= 2*p["N_BATCH"] #7900 #20*p["N_BATCH"] #7756 
p["N_VALIDATE"]= 0 # 256 # p["N_BATCH"] 
p["ETA"]= 0 #1e-2 #5e-3
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.018
p["INPUT_HIDDEN_STD"]= 0.018
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["REGULARISATION"]= True
p["LBD_UPPER"]= 0.00001
p["LBD_LOWER"]= 0.001
p["NU_UPPER"]= 2000
p["NU_LOWER"]= 5
p["WRITE_TO_DISK"]= False

if p["DEBUG"]:
    p["REC_SPIKES"]= ["input", "hidden"]
    p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I")]
    p["REC_SYNAPSES"]= [("hid_to_out", "w")]

print(p)
    
mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
if p["DEBUG"]:
    plt.figure()
    plt.scatter(spike_t["input"],700-spike_ID["input"],s=0.5)
    plt.figure()
    plt.scatter(spike_t["hidden"],spike_ID["hidden"],s=0.5)
    tmax= np.max(spike_t["hidden"])
    t= np.arange(0,tmax,p["TRIAL_MS"])
    v= np.vstack([ np.zeros((1,len(t))), np.ones((1,len(t)))*p["NUM_HIDDEN"]])
    t= np.reshape(t,(1,len(t)))
    t= np.vstack([ t, t ])
    plt.plot(t,v, lw=1)
    plt.figure()
    plt.plot(rec_vars_n["Voutput"][:,:20])
    plt.legend(np.arange(20))
    plt.title("Voutput")
    plt.figure()
    plt.plot(rec_vars_n["Voutput"][:,20:])
    plt.legend(np.arange(20)+20)
    plt.title("Voutput")
    plt.figure()
    plt.plot(rec_vars_n["lambda_Voutput"][:,:20])
    plt.title("lambda_Voutput up to 20")
    plt.figure()
    plt.plot(rec_vars_n["lambda_Voutput"][:,20:])
    plt.title("lambda_Voutput 20 onwards")
    plt.figure()
    plt.plot(rec_vars_n["lambda_Ioutput"][:,:20])
    plt.title("lambda_Voutput up to 20")
    plt.figure()
    plt.plot(rec_vars_n["lambda_Ioutput"][:,20:])
    plt.title("lambda_Voutput 20 onwards")
    print(mn.output.extra_global_params["label"].view[0:2*p["N_BATCH"]])
    plt.figure()
    h, x= np.histogram(spike_ID["hidden"],p["NUM_HIDDEN"])
    h= np.sort(h)
    plt.bar(np.arange(len(h)),np.log(h+2))
mn.hid_to_out.pull_var_from_device("w")
plt.figure()
plt.hist(mn.hid_to_out.vars["w"].view[:],100)
plt.show()
