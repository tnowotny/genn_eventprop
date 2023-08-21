from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import exists

p["TRIAL_MS"]= 1000
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
p["N_TRAIN"]= 7644 
p["N_VALIDATE"]= 512 
p["ETA"]= 5e-3 
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.02 
p["INPUT_HIDDEN_STD"]= 0.01 
p["HIDDEN_OUTPUT_MEAN"]= 0.0 
p["HIDDEN_OUTPUT_STD"]= 0.03 
p["TAU_MEM"] = 20.0
p["TAU_MEM_OUTPUT"]= 20.0
p["TAU_SYN"] = 5.0 
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-11
p["LBD_LOWER"]= 1e-11
p["NU_UPPER"]= 14
p["NU_LOWER"]= 5
p["RHO_UPPER"]= 10000.0100
p["GLB_UPPER"]= 1e-8
p["TIMING"]= True
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["EVALUATION"]= "speaker"

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

p["AUGMENTATION"]= {
    "random_shift": 40.0,
}

#p["REDUCED_CLASSES"]= [0]

p["SPEAKER_LEFT"]= 0
p["COLLECT_CONFUSION"]= False
p["TAU_ACCUMULATOR"]= 5.0

p["HIDDEN_NOISE"]= 0.0

p["DATA_SET"]= "SSC"
p["EVALUATION"]= "validation_set"
p["READ_BUFFERED"]= False
p["DATA_BUFFER_NAME"]= "./data/SSC/mySSC"

print(p)

found= False
i= -1
while not found:
    i=i+1
    jname= os.path.join(p["OUT_DIR"], p["NAME"]+'.'+str(i)+'.json')
    found= not exists(jname)

jfile= open(jname,'w')
json.dump(p,jfile)

mn= SHD_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
#spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.cross_validate_SHD(p)

rname= os.path.join(p["OUT_DIR"], p["NAME"]+'.'+str(i)+'.summary.txt')
sumfile= open(rname,'w')
sumfile.write("Training correct: {}, Valuation correct: {}".format(correct,correct_eval))

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
