from simulator_SHD import p
import os
import json

p["OUT_DIR"]= "scan_JUWELS_16/"
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
p["HIDDEN_OUTPUT_MEAN"]= 1.2
p["HIDDEN_OUTPUT_STD"]= 0.6
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0
p["TAU_MEM_OUTPUT"]= 1000.0
p["TAU_SYN"] = 5.0
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-8 # keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 1e-8
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
p["EVALUATION"]= "speaker"
p["CUDA_VISIBLE_DEVICES"]= True

p["RECURRENT"]= True
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02

p["REWIRE_SILENT"]= True
p["AVG_SNSUM"]= True

# "first_spike" loss function variables
p["TAU_0"]= 2
p["TAU_1"]= 25.0 #6.4
p["ALPHA"]= 1e-3 #3e-3

p["LOSS_TYPE"]= "first_spike_exp"

p["AUGMENTATION"]= {
    "random_shift": 20.0,
}

p["SPEAKER_LEFT"]= 0
p["COLLECT_CONFUSION"]= False
p["TAU_ACCUMULATOR"]= 5.0

lbd= [ 2e-9, 5e-9, 1e-8, 2e-8 ]
tau_0= [ 1, 2, 5 ]
tau_1= [ 20.0, 25.0, 30.0 ]
alpha= [ 5e-4, 1e-3, 2e-3, 5e-3 ]
seeds= [[372, 371],[814,813],[135,134]]

for i in range(4):
    for j in range(3):
        for k in range(3):
            for l in range(4):
                for m in range(2):
                    id= (((i*3+j)*3+k)*4+l)*2+m
                    print(id)
                    p["LBD_UPPER"]=lbd[i]
                    p["LBD_LOWER"]=lbd[i]
                    p["TAU_0"]= tau_0[j]
                    p["TAU_1"]= tau_1[k]
                    p["ALPHA"]= alpha[l]
                    p["TRAIN_DATA_SEED"]= seeds[0][m]
                    p["TEST_DATA_SEED"]= seeds[1][m]
                    p["MODEL_SEED"]= seeds[2][m]        
                    p["NAME"]= "J16_scan_"+str(id)
            
                    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                        json.dump(p, f)
