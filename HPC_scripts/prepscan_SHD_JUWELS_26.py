from simulator_SHD import p
import os
import json

p["DEBUG_HIDDEN_N"]= True
p["OUT_DIR"]= "scan_JUWELS_26/"
p["DT_MS"]= 1
p["BUILD"]= True
p["TIMING"]= False
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["MODEL_SEED"]= 135
p["TRIAL_MS"]= 1400
p["N_MAX_SPIKE"]= 1500
p["N_BATCH"]= 256
p["N_TRAIN"]= 7644 
p["N_VALIDATE"]= 512 
p["N_EPOCH"]= 300
p["SHUFFLE"]= True
p["NUM_HIDDEN"]= 256
p["RECURRENT"]= False
p["TAU_SYN"] = 5.0
p["TAU_MEM"] = 20.0
p["TAU_MEM_OUTPUT"]= 20.0
p["INPUT_HIDDEN_MEAN"]= 0.03
p["INPUT_HIDDEN_STD"]= 0.01
p["HIDDEN_OUTPUT_MEAN"]= 0.0
p["HIDDEN_OUTPUT_STD"]= 0.03
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02
p["PDROP_INPUT"]= 0.0
p["PDROP_HIDDEN"]= 0.0
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-12 # keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 1e-12
p["NU_UPPER"]= 14
p["ETA"]= 5e-3
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["LOAD_LAST"]= False
p["EVALUATION"]= "speaker"
p["CUDA_VISIBLE_DEVICES"]= True
p["AVG_SNSUM"]= True
p["REWIRE_SILENT"]= False
p["AUGMENTATION"]= {}
p["TAU_ACCUMULATOR"]= 5.0
p["HIDDEN_NOISE"]= 0.0

# "first_spike" loss function variables
p["TAU_0"]= 1
p["TAU_1"]= 100.0 #6.4
p["ALPHA"]= 5e-5 #3e-3
p["LOSS_TYPE"]= "max"

# individual experiments with optimised parameters as determined earlier
lbd= [ 2e-10, 5e-10, 1e-9, 2e-9, 5e-9, 1e-8 ]
eta= [ 1e-3, 2e-3 ] 

# sample independently for all of them
rewire= [ False, True ]

for i in range(6):
    for j in range(2):
        for k in range(2):
            id= (i*2+j)*2+k
            print(id)
            p["LBD_UPPER"]=lbd[i]
            p["LBD_LOWER"]=lbd[i]
            p["REWIRE_SILENT"]= rewire[j]
            p["ETA"]= eta[k]
            p["NAME"]= "J26_scan_"+str(id)
                            
            with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                json.dump(p, f)
