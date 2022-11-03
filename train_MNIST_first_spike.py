from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np

p["DATASET"]= "MNIST"
p["TRAIN_DATA_SEED"]= None
p["TEST_DATA_SEED"]= None
p["NAME"]= "test17"
p["NUM_HIDDEN"]= 128
p["HIDDEN_OUTPUT_MEAN"]= 0.9
p["HIDDEN_OUTPUT_STD"]= 0.03
p["N_MAX_SPIKE"]= 500 #120
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.2
p["PDROP_HIDDEN"]= 0.0
p["ADAM_BETA1"]= 0.99
p["ADAM_BETA2"]= 0.9999    
p["N_EPOCH"]= 50
p["N_BATCH"]= 32
p["N_TRAIN"]= 55000
p["ETA"]= 5e-3 #1e-2 #5e-3
p["SHUFFLE"]= True
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["AVG_SNSUM"]= True
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-8
p["LBD_LOWER"]= 1e-8
p["NU_UPPER"]= 4
#p["DEBUG_HIDDEN_N"]= True

# "first_spike" loss function variables
p["TAU_0"]= 1
p["TAU_1"]= 3 #6.4
p["ALPHA"]= 1e-2 #2e-4 #5.63e-2 #3e-3 #2.6e-3

p["LOSS_TYPE"]= "first_spike_exp"
#p["LOSS_TYPE"]= "max"
#p["LOSS_TYPE"]= "sum"
#p["LOSS_TYPE"]= "avg_xentropy"

p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])

p["REC_SPIKES"]= ["input", "hidden","output"]
p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I")]
p["REC_SYNAPSES"]= [("hid_to_out", "w")]

mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
