from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np
import json



with open('scan_1_0_0.json') as f:
    p = json.load(f)


p["N_EPOCH"]= 1
p["ETA"]= 0.0
p["N_TRAIN"]= 1*p["N_BATCH"]
p["REC_SPIKES"]= ["input", "hidden"]
p["REC_NEURONS"]= [("output", "new_max_t"), ("output","V")]
p["LOAD_LAST"]= True
p["WRITE_TO_DISK"]= False
p["DEBUG"]= True
print(p)

mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
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
plt.plot(rec_vars_n["Voutput"])
plt.legend(np.arange(20))
plt.figure()
plt.plot(rec_vars_n["new_max_toutput"])
print(rec_vars_n["new_max_toutput"].shape)
print(mn.output.extra_global_params["label"].view[0:1*p["N_BATCH"]])
plt.figure()
h, x= np.histogram(spike_ID["hidden"],p["NUM_HIDDEN"])
h= np.sort(h)
plt.bar(np.arange(len(h)),np.log(h+2))
mn.hid_to_out.pull_var_from_device("w")
plt.figure()
plt.hist(mn.hid_to_out.vars["w"].view[:],100)
plt.show()
