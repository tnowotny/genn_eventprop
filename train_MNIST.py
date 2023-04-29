from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np

p["DATASET"]= "MNIST"
p["TRAIN_DATA_SEED"]= None
p["TEST_DATA_SEED"]= None
p["NAME"]= "test17"
p["NUM_HIDDEN"]= 128
p["N_MAX_SPIKE"]= 500 #120
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0
p["N_EPOCH"]= 50
p["N_BATCH"]= 32
p["N_TRAIN"]= 55000
p["ETA"]= 1e-2 #5e-3
p["SHUFFLE"]= True
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)

# "first_spike" loss function variables
p["TAU_0"]= 0.5
p["TAU_1"]= 12 #6.4
p["ALPHA"]= 3e-3 #5.63e-2 #3e-3

#p["LOSS_TYPE"]= "first_spike"
#p["LOSS_TYPE"]= "max"
p["LOSS_TYPE"]= "sum"
#p["LOSS_TYPE"]= "avg_xentropy"

p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])


p["REC_SPIKES"]= ["input", "hidden"]
p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I"),("hidden", "V"), ("hidden", "lambda_V"), ("hidden", "lambda_I")]
#p["REC_SYNAPSES"]= [("hid_to_out", "w")]
p["REC_NEURONS_EPOCH_TRIAL"] = [ [0,0], [0,1], [0,2], [1,0]]
p["REC_SPIKES_EPOCH_TRIAL"] = [ [0,0], [0,1], [0,2] ]


mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
plt.figure()
plt.scatter(spike_t["input"], spike_ID["input"],s=0.5)
plt.scatter(spike_t["hidden"], spike_ID["hidden"]+784,s=0.5)
plt.figure()
plt.title("Vhidden")
plt.plot(rec_vars_n["Vhidden"][:,0,:])
plt.figure()
plt.title("lambda_Vhidden")
plt.plot(rec_vars_n["lambda_Vhidden"][:,0,:])
plt.figure()
plt.title("lambda_Ihidden")
plt.plot(rec_vars_n["lambda_Ihidden"][:,0,:])
plt.figure()
plt.title("Voutput")
plt.plot(rec_vars_n["Voutput"][:,0,:])
plt.figure()
plt.title("lambda_Voutput")
plt.plot(rec_vars_n["lambda_Voutput"][:,0,:])
plt.figure()
plt.title("lambda_Ioutput")
plt.plot(rec_vars_n["lambda_Ioutput"][:,0,:])
plt.show()
