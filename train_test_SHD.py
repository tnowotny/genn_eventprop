from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json

p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["NAME"]= "tt1"
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 1500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.1
p["PDROP_HIDDEN"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["DEBUG"]= False
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 2
p["N_BATCH"]= 256
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 8156 # that is all of them
p["N_VALIDATE"]= 0 # no validation
p["ETA"]= 5e-3 #5e-3
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.02 # 0.02
p["INPUT_HIDDEN_STD"]= 0.01 # 0.01
p["HIDDEN_OUTPUT_MEAN"]= 0.0
p["HIDDEN_OUTPUT_STD"]= 0.03 # 0.3
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0 #20
p["TAU_SYN"] = 5.0 #5
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-12 # 5e-12 keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 1e-5
p["NU_UPPER"]= 15 #*p["N_BATCH"]
p["NU_LOWER"]= 5
p["RHO_UPPER"]= 10000.0
p["GLB_UPPER"]= 1e-8
p["ETA_DECAY"]= 1.0      
p["ETA_FIDDELING"]= False
p["ETA_REDUCE"]= 0.5
p["ETA_REDUCE_PERIOD"]= 50
p["TIMING"]= False
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["LOSS_TYPE"]= "sum"
p["EVALUATION"]= "random"

p["RECURRENT"]= True
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02 # 0.02

p["REWIRE_SILENT"]= True
p["AVG_SNSUM"]= True

p["AUGMENTATION"]= {
    "random_shift": 20.0,
    "random_dilate": (0.9, 1.1),
    "ID_jitter": 5.0
}

if p["DEBUG"]:
    p["REC_SPIKES"]= ["input", "hidden"]
    #p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I")]
    #p["REC_SYNAPSES"]= [("hid_to_out", "w")]

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as file:
    json.dump(p, file)

p["N_TRAIN"]= 8156
p["N_VALIDATE"]= 0
for i in range(10):
    mn= SHD_model(p)
    spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train_test(p)
    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_traintest.txt'),'a') as f:
        f.write("{} {}\n".format(correct,correct_eval))
    p["TRAIN_DATA_SEED"]+= 31
