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
p["PDROP_INPUT"]= 0.2
p["ADAM_BETA1"]= 0.99
p["ADAM_BETA2"]= 0.9999    
p["N_EPOCH"]= 50
p["N_BATCH"]= 32
p["N_TRAIN"]= 55000
p["ETA"]= 1e-2 #5e-3
p["SHUFFLE"]= True

#p["LOSS_TYPE"]= "first_spike"
#p["LOSS_TYPE"]= "max"
#p["LOSS_TYPE"]= "sum"
p["LOSS_TYPE"]= "avg_xentropy"

mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))

