from simulator_MNIST import p
import os
import json

p["OUT_DIR"]= "scan_JUWELS_1/"
p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 3000
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.1
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
p["HIDDEN_OUTPUT_STD"]= 0.3
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-13 # keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 1e-13
p["NU_UPPER"]= 10
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

nup= [ 10, 15, 20 ]
eta= [ 2e-3, 5e-3, 1e-2 ]
lup= [ 1e-12, 5e-12 ]
btch= [ 32, 256 ]

for i in range(3):
    for j in range(3):
        for k in range(2):
            for l in range(2):
                id= ((i*3+j)*2+k)*2+l
                p["NU_UPPER"]= nup[i]
                p["ETA"]= eta[j]
                p["LBD_UPPER"]= lup[k]
                p["N_BATCH"]= btch[l] 
                p["NAME"]= "scan_"+str(id)
                
                with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                    json.dump(p, f)
    

