from simulator_SHD import p
import os
import json
import numpy as np
    
p["TRIAL_MS"]= 280
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
p["ETA"]= 5e-3
p["SHUFFLE"]= True
p["RECURRENT"]= True
p["INPUT_HIDDEN_MEAN"]= 0.2 
p["INPUT_HIDDEN_STD"]= 0.1 
p["HIDDEN_OUTPUT_MEAN"]= 0.0 
p["HIDDEN_OUTPUT_STD"]= 0.03 
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02 # 0.02
p["HIDDEN_HIDDENFWD_MEAN"]= 0.04 # only used when > 1 hidden layer
p["HIDDEN_HIDDENFWD_STD"]= 0.01 # only used when > 1 hidden layer
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
p["RESCALE_T"]= 0.2
p["RESCALE_X"]= 0.1

p["DEBUG_HIDDEN_N"]= True
p["OUT_DIR"]= "scan_JUWELS_30/"
p["BUILD"]= True

p["TRAIN_TAUM"]= False
p["N_HID_LAYER"]= 2

# learning rate schedule depending on exponential moving average of performance
p["EMA_ALPHA1"]= 0.2
p["EMA_ALPHA2"]= 0.05
p["ETA_FAC"]= 0.5
p["MIN_EPOCH_ETA_FIXED"]= 20

# sample a lot of stuff
train_taum= [ False, True ]
n_hid_layer= [ 1, 2 ]
num_hidden= [ 256, 1024 ]
scale_x= [ 0.1, 0.2, 1.0 ]
scale_t= [ 0.1, 0.2, 1.0 ]
n_batch= [ 32, 256 ]
lbd= [ 1e-7, 2e-7, 5e-7 ]
nu= [ 7, 14 ]
rec= [False, True]

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(3):
                for m in range(3):
                    for n in range(2):
                        for o in range(3):
                            for q in range(2):
                                for r in range(2):
                                    id= (((((((i*2+j)*2+k)*3+l)*3+m)*2+n)*3+o)*2+q)*2+r
                                    print(id)
                                    p["TRAIN_TAUM"]= train_taum[i]
                                    p["N_HID_LAYER"]= n_hid_layer[j]
                                    p["NUM_HIDDEN"]= num_hidden[k]
                                    p["HIDDEN_HIDDEN_STD"]= 0.02/num_hidden[k]*256
                                    p["HIDDEN_HIDDENFWD_MEAN"]= 0.1/num_hidden[k]*256 
                                    p["HIDDEN_HIDDENFWD_STD"]= 0.02/num_hidden[k]*256
                                    p["RESCALE_X"]= scale_x[l]
                                    p["INPUT_HIDDEN_MEAN"]= 0.02/scale_x[l] 
                                    p["INPUT_HIDDEN_STD"]= 0.01/scale_x[l]
                                    p["AUGMENTATION"]= {
                                        "random_shift": 40.0*scale_x[l],
                                    }
                                    p["RESCALE_T"]= scale_t[m]
                                    p["TRIAL_MS"]= 1400*scale_t[m]
                                    p["TAU_MEM"] = 20.0*np.sqrt(scale_t[m])
                                    p["TAU_SYN"] = 5.0*np.sqrt(scale_t[m])
                                    p["N_BATCH"]= n_batch[n]
                                    p["LBD_UPPER"]= lbd[o]
                                    p["LBD_LOWER"]= lbd[o]
                                    p["NU_UPPER"]= nu[q]*np.sqrt(scale_t[m])
                                    p["NU_LOWER"]= nu[q]*np.sqrt(scale_t[m])
                                    p["RECURRENT"]= rec[r]
                                    p["NAME"]= "J30_scan_"+str(id)
                                    
                                    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                        json.dump(p, f)
