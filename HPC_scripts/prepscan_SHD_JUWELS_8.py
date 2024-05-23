from simulator_SHD import p
import os
import json

p["OUT_DIR"]= "scan_JUWELS_8/"
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
p["N_EPOCH"]= 200
p["N_BATCH"]= 32
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 7900 #20*p["N_BATCH"] #7756 
p["N_VALIDATE"]= 512 # 256 # p["N_BATCH"] 
p["ETA"]= 1e-3
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.02
p["INPUT_HIDDEN_STD"]= 0.01
p["HIDDEN_OUTPUT_MEAN"]= 0.0
p["HIDDEN_OUTPUT_STD"]= 0.3
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 2e-9 # keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 2e-9
p["NU_UPPER"]= 14
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
    "random_shift": 20.0,
    "random_dilate": (0.9, 1.1),
    "ID_jitter": 5.0
}

n_batch= [ 32, 64, 128 ]
lbd= [ 5e-15, 1e-14, 2e-14 ]
shift= [ 0.0, 10.0, 20.0 ]
dilate= [ 0.0, 0.1, 0.2 ]
jitter= [ 0.0, 5.0]
seeds= [[372, 371],[814,813],[135,134]]

for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                for m in range(2):
                    for n in range(2):
                        id= ((((i*3+j)*3+k)*3+l)*2+m)*2+n
                        p["N_BATCH"]= n_batch[i]
                        p["LBD_UPPER"]=lbd[j]
                        p["LBD_LOWER"]=lbd[j]
                        p["AUGMENTATION"]["random_shift"]= shift[k]
                        p["AUGMENTATION"]["random_dilate"]= (1-dilate[l],1+dilate[l])
                        p["AUGMENTATION"]["ID_jitter"]= jitter[m]
                        p["TRAIN_DATA_SEED"]= seeds[0][n]
                        p["TEST_DATA_SEED"]= seeds[1][n]
                        p["MODEL_SEED"]= seeds[2][n]        
                        p["NAME"]= "scan_"+str(id)
            
                        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                            json.dump(p, f)
