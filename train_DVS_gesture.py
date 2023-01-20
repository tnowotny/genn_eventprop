from simulator_DVS_gesture import *
import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import exists

p["TRIAL_MS"]= 18000
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["MODEL_SEED"]= 135
p["NAME"]= "test106"
p["NUM_HIDDEN"]= 512
p["N_MAX_SPIKE"]= 18000
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.1
p["PDROP_HIDDEN"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 300
p["N_BATCH"]= 32
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 1045
p["N_VALIDATE"]= 32
p["ETA"]= 5e-3 
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.008 
p["INPUT_HIDDEN_STD"]= 0.004
p["HIDDEN_OUTPUT_MEAN"]= 0.0 
p["HIDDEN_OUTPUT_STD"]= 0.03 
p["TAU_MEM"] = 20.0
p["TAU_MEM_OUTPUT"]= 20.0
p["TAU_SYN"] = 5.0 
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 2e-10 #2e-9
p["LBD_LOWER"]= 2e-10 #2e-9
p["NU_UPPER"]= 100
p["NU_LOWER"]= 5
p["RHO_UPPER"]= 10000.0100
p["GLB_UPPER"]= 1e-8
p["TIMING"]= True
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])//90
print(p["SPK_REC_STEPS"])

p["RECURRENT"]= True
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02 # 0.02

p["REWIRE_SILENT"]= True
p["AVG_SNSUM"]= True

# "first_spike" loss function variables
p["TAU_0"]= 2
p["TAU_1"]= 25.0
p["ALPHA"]= 1e-3 

p["LOSS_TYPE"]= "sum" #"sum_weigh_exp"

p["AUGMENTATION"]= {}

#p["REDUCED_CLASSES"]= [0]

p["COLLECT_CONFUSION"]= False
p["TAU_ACCUMULATOR"]= 5.0

p["HIDDEN_NOISE"]= 0.0
    
found= False
i= -1
while not found:
    i=i+1
    jname= os.path.join(p["OUT_DIR"], p["NAME"]+'.'+str(i)+'.json')
    found= not exists(jname)

jfile= open(jname,'w')
json.dump(p,jfile)

mn= DVSG_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
#spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.cross_validate_SHD(p)

rname= os.path.join(p["OUT_DIR"], p["NAME"]+'.'+str(i)+'.summary.txt')
sumfile= open(rname,'w')
sumfile.write("Training correct: {}, Valuation correct: {}".format(correct,correct_eval))

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
