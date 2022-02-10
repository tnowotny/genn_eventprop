from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np

p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= None
p["TEST_DATA_SEED"]= None
p["NAME"]= "test17"
p["NUM_HIDDEN"]= 512
p["N_MAX_SPIKE"]= 500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.0
p["ADAM_BETA1"]= 0.999
p["ADAM_BETA2"]= 0.99999   
p["DEBUG"]= False
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 300
p["N_BATCH"]= 128
p["N_TRAIN"]= 7808 #20*p["N_BATCH"] #7756 
p["N_VALIDATE"]= 256 #p["N_BATCH"] 
p["ETA"]= 1e-2 #5e-3
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

if p["DEBUG"]:
    p["REC_SPIKES"]= ["input", "hidden"]
    p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I")]
    p["REC_SYNAPSES"]= [("hid_to_out", "w")]
mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
if p["DEBUG"]:
    plt.figure()
    plt.scatter(spike_t["input"],700-spike_ID["input"],s=0.5)
    plt.figure()
    plt.scatter(spike_t["hidden"],spike_ID["hidden"],s=0.5)
    plt.figure()
    plt.plot(rec_vars_n["Voutput"])
    plt.figure()
    plt.plot(rec_vars_n["lambda_Voutput"])
    plt.figure()
    plt.plot(rec_vars_n["lambda_Ioutput"])
    print(mn.output.extra_global_params["label"].view[0:20*p["N_BATCH"]:p["N_BATCH"]])
mn.hid_to_out.pull_var_from_device("w")
plt.figure()
plt.hist(mn.hid_to_out.vars["w"].view[:],100)
plt.show()
