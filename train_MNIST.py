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
#p["LOSS_TYPE"]= "sum"
p["LOSS_TYPE"]= "avg_xentropy"

#p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])


#p["REC_SPIKES"]= ["input", "hidden","output"]
#p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I")]
#p["REC_SYNAPSES"]= [("hid_to_out", "w")]

#p["REC_SPIKES_EPOCH_TRIAL"] = [ [0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [20,0], [20,1], [20,2], [90,0], [90,1], [90,2]  ]


mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))

