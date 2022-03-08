from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np
import json

p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["NAME"]= "test20"
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.0
p["ADAM_BETA1"]= 0.99
p["ADAM_BETA2"]= 0.9999   
p["DEBUG"]= False
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 500
p["N_BATCH"]= 128
p["N_TRAIN"]= 7900 #20*p["N_BATCH"] #7756 
p["N_VALIDATE"]= 256 # 256 # p["N_BATCH"] 
p["ETA"]= 5e-3 #1e-2 #5e-3
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.01
p["INPUT_HIDDEN_STD"]= 0.01
p["HIDDEN_OUTPUT_MEAN"]= 0.1
p["HIDDEN_OUTPUT_STD"]= 0.3
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["REGULARISATION"]= True
p["LBD_UPPER"]= 0.0000001 # keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 0.005
p["NU_UPPER"]= 17*p["N_BATCH"]
p["NU_LOWER"]= 0.1*p["N_BATCH"]
p["RHO_UPPER"]= 5000.0
p["GLB_UPPER"]= 0.0
p["ETA_DECAY"]= 1.0      
p["WRITE_TO_DISK"]= True

hos= [ 0.1, 0.2, 0.3, 0.4, 0.5 ]
ihs= [ 0.005, 0.01, 0.015, 0.02 ]

cc= [ [] for _ in range(4) ]
cce= [ [] for _ in range(4) ]
for i in range(4):
    for j in range(4):
        p["HIDDEN_OUTPUT_STD"]= hos[i]
        p["INPUT_HIDDEN_STD"]= ihs[j]
        p["NAME"]= "scan_"+str(i)+"_"+str(j)
        
        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as file:
            json.dump(p, file)
    
        mn= mnist_model(p)
        spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
        print("correct: {}".format(correct))
        print("correct_eval: {}".format(correct_eval))
        cc[i].append(correct)
        cce[i].append(correct_eval)

print(cc)
print(cce)
