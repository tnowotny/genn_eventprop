from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np

p["TRAIN_DATA_SEED"]= None
p["TEST_DATA_SEED"]= None
p["NAME"]= "MNIST_first_spike"
p["OUT_DIR"]= "test_MNIST_simple"
p["NUM_HIDDEN"]= 128
p["HIDDEN_OUTPUT_MEAN"]= 0.9
p["HIDDEN_OUTPUT_STD"]= 0.03
p["N_MAX_SPIKE"]= 500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.2
p["ADAM_BETA1"]= 0.99
p["ADAM_BETA2"]= 0.9999    
p["N_EPOCH"]= 50
p["N_BATCH"]= 32
p["N_TRAIN"]= 55000
p["ETA"]= 5e-3
p["SHUFFLE"]= True
p["AVG_SNSUM"]= True
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-8
p["LBD_LOWER"]= 1e-8
p["NU_UPPER"]= 4

# "first_spike" loss function variables
p["TAU_0"]= 1
p["TAU_1"]= 3
p["ALPHA"]= 3.6e-4

p["LOSS_TYPE"]= "first_spike_exp"

mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
