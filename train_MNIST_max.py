from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np
import json

p["TRAIN_DATA_SEED"]= None
p["TEST_DATA_SEED"]= None
p["NAME"]= "MNIST_max"
p["OUT_DIR"]= "test_MNIST_simple"
p["NUM_HIDDEN"]= 128
p["N_MAX_SPIKE"]= 500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.2
p["N_EPOCH"]= 50
p["N_BATCH"]= 32
p["N_TRAIN"]= 55000
p["ETA"]= 1e-2
p["SHUFFLE"]= True

p["LOSS_TYPE"]= "max"

jname= os.path.join(p["OUT_DIR"], p["NAME"]+'.json')
jfile= open(jname,'w')
json.dump(p,jfile)

mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
