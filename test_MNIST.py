from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np

p["NAME"]= "test14"
p["NUM_HIDDEN"]= 700
p["N_MAX_SPIKE"]= 60
p["DT_MS"]= 1
p["DEBUG"]= False
p["N_BATCH"]= 500
p["SHUFFLE"]= True
p["LOAD_LAST"]= True

if p["DEBUG"]:
    p["REC_SPIKES"]= ["input", "hidden"]
    p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I")]
    p["REC_SYNAPSES"]= [("hid_to_out", "w")]
mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.test(p)
