from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np

p["ADAM_BETA1"]= 0.95      
p["ADAM_BETA2"]= 0.9995    
p["DEBUG"]= False
p["N_EPOCH"]= 1000
p["N_BATCH"]= 5
p["N_TRAIN"]= 55000
p["ETA"]= 5e-3
p["SHUFFLE"]= False
if p["DEBUG"]:
    p["REC_SPIKES"]= ["input", "hidden"]
    p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I")]
    p["REC_SYNAPSES"]= [("hid_to_out", "w")]
mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct= mn.train(p)

print("correct: {}".format(correct))
if p["DEBUG"]:
    plt.figure()
    plt.scatter(spike_t["input"],spike_ID["input"],s=0.5)
    plt.figure()
    plt.scatter(spike_t["hidden"],spike_ID["hidden"],s=0.5)
    plt.figure()
    plt.plot(rec_vars_n["Voutput"])
    plt.figure()
    plt.plot(rec_vars_n["lambda_Voutput"])
    plt.figure()
    plt.plot(rec_vars_n["lambda_Ioutput"])
    plt.figure()
    plt.subplot(2,1,1)
    tlbl=  mn.output.extra_global_params["label"].view[0]
    msk= np.zeros(rec_vars_s["whid_to_out"].shape[1], dtype= bool)
    print(msk.shape)
    msk[tlbl::NUM_OUTPUT]= True
    plt.plot(rec_vars_s["whid_to_out"][:,msk])
    plt.subplot(2,1,2)
    plt.plot(rec_vars_s["whid_to_out"][:,np.logical_not(msk)])

    sT= spike_t["input"]
    sID= spike_ID["input"]
    for i in range(4):
        plt.figure()
        plt.subplot(1,2,1)
        x= np.zeros(NUM_INPUT)
        msk= np.logical_and(sT > i*20.0, sT < (i+1)*20.0)
        x[sID[msk]]= 20.0-(sT[msk]-i*20.0)
        plt.imshow(x.reshape((28,28)))
        plt.subplot(1,2,2)
        plt.imshow(mn.X_train_orig[i*p["N_BATCH"],:,:])
    print(mn.output.extra_global_params["label"].view[0:4*p["N_BATCH"]:p["N_BATCH"]])
mn.hid_to_out.pull_var_from_device("w")
plt.figure()
plt.hist(mn.hid_to_out.vars["w"].view[:],100)
#plt.show()
