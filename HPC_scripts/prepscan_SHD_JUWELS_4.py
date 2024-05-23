from simulator_MNIST import p
import os
import json

p["OUT_DIR"]= "scan_JUWELS_4/"
p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 3000
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.1
p["PDROP_HIDDEN"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["DEBUG"]= False
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 300
p["N_BATCH"]= 256
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 7900 #20*p["N_BATCH"] #7756 
p["N_VALIDATE"]= 512 # 256 # p["N_BATCH"] 
p["ETA"]= 5e-3
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.02
p["INPUT_HIDDEN_STD"]= 0.01
p["HIDDEN_OUTPUT_MEAN"]= 0.0
p["HIDDEN_OUTPUT_STD"]= 0.03
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-12 # keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 1e-13
p["NU_UPPER"]= 15
p["NU_LOWER"]= 1*p["N_BATCH"]
p["RHO_UPPER"]= 5000.0
p["GLB_UPPER"]= 0.001
p["ETA_DECAY"]= 1.0      
p["ETA_FIDDELING"]= False
p["ETA_REDUCE"]= 0.5
p["ETA_REDUCE_PERIOD"]= 50
p["TIMING"]= False
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["LOSS_TYPE"]= "sum"
p["EVALUATION"]= "speaker"
p["CUDA_VISIBLE_DEVICES"]= True

p["RECURRENT"]= True
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02

p["REWIRE_SILENT"]= True
p["AVG_SNSUM"]= True

p["AUGMENTATION"]= {
    "random_shift": 0.0,
    "random_dilate": (0.9, 1.1),
    "ID_jitter": 0.0
}

shift= [ 0.0, 5.0, 10.0, 20.0 ]
jitter= [0.0, 5.0, 10.0, 20.0 ]
tau_m= [ 10.0, 20.0, 20.0, 30.0 ]
tau_s= [ 5.0, 5.0, 10.0, 15.0 ]

for j in range(4):
    for k in range(4):
        for l in range(4):
            id= (j*4+k)*4+l
            p["AUGMENTATION"]["random_shift"]= shift[j]
            p["AUGMENTATION"]["ID_jitter"]= jitter[k]
            p["TAU_MEM"]= tau_m[l]
            p["TAU_SYN"]= tau_s[l]
            p["NAME"]= "scan_"+str(id)
            
            with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                json.dump(p, f)
    

