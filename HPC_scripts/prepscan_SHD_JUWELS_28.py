from simulator_SHD import p
import os
import json

p["TRIAL_MS"]= 140
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["MODEL_SEED"]= 135
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 1500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.0
p["PDROP_HIDDEN"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 300
p["N_BATCH"]= 32
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 7644 
p["N_VALIDATE"]= 512 
p["ETA"]= 1e-2 
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.2 
p["INPUT_HIDDEN_STD"]= 0.1 
p["HIDDEN_OUTPUT_MEAN"]= 0.0 
p["HIDDEN_OUTPUT_STD"]= 0.03 
p["TAU_MEM"] = 8.0
p["TAU_MEM_OUTPUT"]= 8.0
p["TAU_SYN"] = 2.0 
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-6
p["LBD_LOWER"]= 1e-6
p["NU_UPPER"]= 3
p["NU_LOWER"]= 3
p["RHO_UPPER"]= 10000.0100
p["GLB_UPPER"]= 1e-8
p["TIMING"]= True
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["EVALUATION"]= "speaker"

p["RECURRENT"]= True
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02 # 0.02

p["REWIRE_SILENT"]= True
p["REWIRE_LIFT"]= 0.01
p["AVG_SNSUM"]= True

# "first_spike" loss function variables
p["TAU_0"]= 2
p["TAU_1"]= 25.0
p["ALPHA"]= 1e-3 

p["LOSS_TYPE"]= "sum_weigh_exp"

p["AUGMENTATION"]= {
    "random_shift": 4.0,
}

p["SPEAKER_LEFT"]= 0
p["COLLECT_CONFUSION"]= False
p["TAU_ACCUMULATOR"]= 5.0

p["HIDDEN_NOISE"]= 0.0
p["RESCALE_T"]= 0.1
p["RESCALE_X"]= 0.1

p["DEBUG_HIDDEN_N"]= True
p["OUT_DIR"]= "scan_JUWELS_28/"
p["BUILD"]= True

# sample a lot of stuff
eta= [ 5e-3, 1e-2 ] 
n_batch= [ 32, 256 ]
drop_input= [ 0, 0.05, 0.1 ]
lbd= [ 1e-8, 1e-7, 1e-6, 1e-5 ]
nu= [ 1, 2, 3, 4 ]

for i in range(2):
    for j in range(2):
        for k in range(3):
            for l in range(4):
                for m in range(4):
                    id= (((i*2+j)*3+k)*4+l)*4+m
                    print(id)
                    p["ETA"]= eta[i]
                    p["N_BATCH"]= n_batch[j]
                    p["PDROP_INPUT"]= drop_input[k]
                    p["LBD_UPPER"]= lbd[l]
                    p["LBD_LOWER"]= lbd[l]
                    p["NU_UPPER"]= nu[m]
                    p["NU_LOWER"]= nu[m]
                    p["NAME"]= "J28_scan_"+str(id)
                            
                    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                    json.dump(p, f)
