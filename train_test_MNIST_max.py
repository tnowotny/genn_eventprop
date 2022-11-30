from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np
import json

p["NAME"]= "MNIST_max"
p["OUT_DIR"]= "test_MNIST_all"
p["NUM_HIDDEN"]= 128
p["N_MAX_SPIKE"]= 500 
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.2
p["N_EPOCH"]= 50
p["N_BATCH"]= 32
p["ETA"]= 1e-2 
p["SHUFFLE"]= True

p["LOSS_TYPE"]= "max"

jname= os.path.join(p["OUT_DIR"], p["NAME"]+'.json')
jfile= open(jname,'w')
json.dump(p,jfile)

p["N_TRAIN"]= 55000
p["N_VALIDATE"]= 5000
for i in range(10):
    mn= mnist_model(p)
    spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_validation.txt'),'a') as f:
        f.write("{} {}\n".format(correct,correct_eval))
    p["TRAIN_DATA_SEED"]+= 31

p["N_TRAIN"]= 60000
p["N_VALIDATE"]= 0
for i in range(10):
    mn= mnist_model(p)
    spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train_test(p)
    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_traintest.txt'),'a') as f:
        f.write("{} {}\n".format(correct,correct_eval))
    p["TRAIN_DATA_SEED"]+= 31
