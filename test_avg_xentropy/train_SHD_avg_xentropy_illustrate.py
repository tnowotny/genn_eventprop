from simulator_MNIST import *
import matplotlib.pyplot as plt
import numpy as np
import json

p["TRIAL_MS"]= 400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["MODEL_SEED"]= 135
p["NAME"]= "test_axe2"
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
p["N_EPOCH"]= 10
p["N_BATCH"]= 32
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 256 #7900
p["N_VALIDATE"]= 32
p["ETA"]= 2e-3 
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.02 
p["INPUT_HIDDEN_STD"]= 0.01 
p["HIDDEN_OUTPUT_MEAN"]= 0.0 
p["HIDDEN_OUTPUT_STD"]= 0.3 
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0 
p["TAU_SYN"] = 5.0 
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 0
p["LBD_LOWER"]= 2e-8 #2e-8
p["NU_UPPER"]= 15 
p["NU_LOWER"]= 5
p["RHO_UPPER"]= 10000.0
p["GLB_UPPER"]= 1e-8
p["ETA_DECAY"]= 1.0      
p["ETA_FIDDELING"]= False
p["ETA_REDUCE"]= 0.5
p["ETA_REDUCE_PERIOD"]= 50
p["TIMING"]= True
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["LOSS_TYPE"]= "avg_xentropy"
p["EVALUATION"]= "speaker"

p["RECURRENT"]= False
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02 # 0.02

p["REWIRE_SILENT"]= False
p["AVG_SNSUM"]= True

p["AUGMENTATION"]= {}

p["REDUCED_CLASSES"]= [0]

p["REC_NEURONS"]= [("output","V"),("output","sum_V"),("output","lambda_V"),("output","lambda_I"),
                   ("hidden","lambda_V")]
#p["REC_NEURONS_EPOCH_TRIAL"]= [(0,25),(0,26),(0,27),(0,28),(0,29),(0,30),(0,31),
#                               (9,25),(9,26),(9,27),(9,28),(9,29),(9,30),(9,31),
#]

p["REC_NEURONS_EPOCH_TRIAL"]= [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
                               (9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),
]


#p["REC_SPIKES_EPOCH_TRIAL"]= [(0,25),(0,26),(0,27),(0,28),(0,29),(0,30),(0,31),
#                              (9,25),(9,26),(9,27),(9,28),(9,29),(9,30),(9,31),
#]
p["REC_SPIKES"]= ["input","hidden"]
p["REC_SPIKES_EPOCH_TRIAL"]= [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
                               (9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),
]

#p["W_OUTPUT_EPOCH_TRIAL"]= [(0,25),(0,26),(0,27),(0,28),(0,29),(0,30),(0,31),
#                              (9,25),(9,26),(9,27),(9,28),(9,29),(9,30),(9,31),
#]
p["W_OUTPUT_EPOCH_TRIAL"]= [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
                               (9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),
]

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as file:
    json.dump(p, file)
    
mn= mnist_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
