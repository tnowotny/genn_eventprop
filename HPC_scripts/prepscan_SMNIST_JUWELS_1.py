from simulator_SMNIST import p
import os
import json

p["OUT_DIR"]= "scan_SMNIST_JU_1/"
p["TRIAL_MS"]= 1500
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["MODEL_SEED"]= 135
p["NAME"]= "test105"
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 1500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.1
p["PDROP_HIDDEN"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 300
p["N_BATCH"]= 256
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 55000
p["N_VALIDATE"]= 5000 
p["ETA"]= 2e-3 
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.02 
p["INPUT_HIDDEN_STD"]= 0.01 
p["HIDDEN_OUTPUT_MEAN"]= 0.0 
p["HIDDEN_OUTPUT_STD"]= 0.03 
p["TAU_MEM"] = 20.0
p["TAU_MEM_OUTPUT"]= 20.0
p["TAU_SYN"] = 5.0 
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 2e-9
p["LBD_LOWER"]= 2e-9
p["NU_UPPER"]= 20
p["NU_LOWER"]= 5
p["RHO_UPPER"]= 10000.0100
p["GLB_UPPER"]= 1e-8
p["TIMING"]= True
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])

p["RECURRENT"]= True
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02 # 0.02

p["REWIRE_SILENT"]= True
p["AVG_SNSUM"]= True

# "first_spike" loss function variables
p["TAU_0"]= 2
p["TAU_1"]= 25.0
p["ALPHA"]= 1e-3 

p["LOSS_TYPE"]= "sum_weigh_exp"

p["AUGMENTATION"]= {}

p["COLLECT_CONFUSION"]= False
p["TAU_ACCUMULATOR"]= 5.0

p["HIDDEN_NOISE"]= 0.0

# individual experiments with optimised parameters as determined earlier
n_hid= [ 128, 256, 512 ]
n_batch= [ 32, 64, 128, 256 ]
lbd= [ 1e-9, 2e-9, 5e-9, 1e-8 ]
nu= [ 15, 20, 25, 30 ]
eta= [ 1e-3, 2e-3, 5e-3 ] 
rewire= [ False, True ]
recurrent= [ False, True ]
pdrop_input= [ 0.0, 0.1 ]
seeds= [[372, 371],[814,813],[135,134]]

for i in range(3):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                for m in range(3):
                    for n in range(2):
                        for n2 in range(2):
                            for n3 in range(2):
                                for o in range(2):
                                    id= (((((((i*4+j)*4+k)*4+l)*3+m)*2+n)*2+n2)*2+n3)*2+o
                                    print(id)
                                    p["NUM_HIDDEN"]= n_hid[i]
                                    p["N_BATCH"]= n_batch[j]
                                    p["LBD_UPPER"]= lbd[k]
                                    p["LBD_LOWER"]= lbd[k]
                                    p["NU_UPPER"]= nu[l]
                                    p["ETA"]= eta[m]
                                    p["REWIRE_SILENT"]= rewire[n]
                                    p["RECURRENT"]= recurrent[n2]
                                    p["PDROP_INPUT"]= pdrop_input[n3]
                                    p["TRAIN_DATA_SEED"]= seeds[0][o]
                                    p["TEST_DATA_SEED"]= seeds[1][o]
                                    p["MODEL_SEED"]= seeds[2][o]
                                    p["NAME"]= "SMJ1_scan_"+str(id)
                            
                                    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                        json.dump(p, f)
