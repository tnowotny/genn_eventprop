from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np

# record some spikes and neuron variables
p["REC_SPIKES"]= ["input", "hidden"]
p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I"),("hidden", "V"), ("hidden", "lambda_V"), ("hidden", "lambda_I")]
p["REC_NEURONS_EPOCH_TRIAL"] = [ [0,0], [0,1], [0,2], [1,0]]
p["REC_SPIKES_EPOCH_TRIAL"] = [ [0,0], [0,1], [0,2] ]

# Adjust some settings here as desired
p["LOSS_TYPE"] = "avg_xentropy"

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
