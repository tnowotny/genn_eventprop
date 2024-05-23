from simulator_SHD import p
import os
import json

p["DEBUG_HIDDEN_N"]= True
p["OUT_DIR"]= "scan_JUWELS_27/"
p["DT_MS"]= 1
p["BUILD"]= True
p["TIMING"]= False
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["MODEL_SEED"]= 135
p["TRIAL_MS"]= 140
p["N_MAX_SPIKE"]= 1500
p["N_BATCH"]= 256
p["N_TRAIN"]= 7644 
p["N_VALIDATE"]= 512 
p["N_EPOCH"]= 300
p["SHUFFLE"]= True
p["NUM_HIDDEN"]= 256
p["RECURRENT"]= True
p["TAU_SYN"] = 5.0
p["TAU_MEM"] = 20.0
p["TAU_MEM_OUTPUT"]= 20.0
p["INPUT_HIDDEN_MEAN"]= 0.2
p["INPUT_HIDDEN_STD"]= 0.1
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
p["REWIRE_SILENT"]= True
p["AUGMENTATION"]= {}
p["TAU_ACCUMULATOR"]= 5.0
p["HIDDEN_NOISE"]= 0.0
p["LOSS_TYPE"]= "sum_weigh_exp"
# "first_spike" loss function variables
p["TAU_0"]= 1
p["TAU_1"]= 100.0 #6.4
p["ALPHA"]= 5e-5 #3e-3
p["RESCALE_T"]= 0.1
p["RESCALE_X"]= 0.1

# sample a lot of stuff
tausyn= [ 0.2, 0.5, 1.0, 2.0 ]
taumem= [ 1.0, 2.0, 4.0, 8.0 ]
drop_input= [ 0, 0.05, 0.1 ]
n_batch= [ 32, 256 ]
lbd= [ 1e-9, 1e-8, 1e-7, 1e-6 ]
eta= [ 5e-3, 1e-2 ] 
augment= [{}, {
    "random_shift": 4.0,
}]
recurrent= [ False, True ]

for i in range(4):
    for j in range(4):
        for k in range(3):
            for l in range(2):
                for m in range(4):
                    for n in range(2):
                        for o in range(2):
                            for q in range(2):
                                id= (((((((i*4+j)*3+k)*2+l)*4+m)*2+n)*2+o)*2+q)
                                print(id)
                                p["TAU_SYN"]= tausyn[i]
                                p["TAU_MEM"]= taumem[j]
                                p["TAU_MEM_OUTPUT"]= taumem[j]
                                p["PDROP_INPUT"]= drop_input[k]
                                p["N_BATCH"]= n_batch[l]
                                p["LBD_UPPER"]= lbd[m]
                                p["LBD_LOWER"]= lbd[m]
                                p["ETA"]= eta[n]
                                p["AUGMENTATION"]= augment[o]
                                p["RECURRENT"]= recurrent[q]
                                p["NAME"]= "J27_scan_"+str(id)
                            
                                with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                                    json.dump(p, f)
