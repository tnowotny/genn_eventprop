from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import exists

p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["MODEL_SEED"]= 135
p["NAME"]= "test102"
p["NUM_HIDDEN"]= 128
p["N_MAX_SPIKE"]= 1500
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
p["ETA"]= 1e-3 #1e-3 1e-4 # 5e-3
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.02 # 0.02
p["INPUT_HIDDEN_STD"]= 0.01 # 0.01
p["HIDDEN_OUTPUT_MEAN"]= 0.0 # 0.06 0.0
p["HIDDEN_OUTPUT_STD"]= 0.3 # 0.03 0.3
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0 #20
p["TAU_MEM_OUTPUT"]= 20.0
p["TAU_SYN"] = 5.0 #5
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 2e-9 # 2e-9 # 2e-8 # 2e-14 (since removal of N_Batch), 5e-12 keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 2e-9 #2e-8
p["NU_UPPER"]= 14
p["NU_LOWER"]= 5
p["RHO_UPPER"]= 10000.0100
p["GLB_UPPER"]= 1e-8
p["ETA_DECAY"]= 1.0      
p["ETA_FIDDELING"]= False
p["ETA_REDUCE"]= 0.5
p["ETA_REDUCE_PERIOD"]= 50
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
p["TAU_1"]= 25.0 #6.4
p["ALPHA"]= 1e-3 #3e-3

p["LOSS_TYPE"]= "sum_weigh_exp"
#p["LOSS_TYPE"]= "max"
#p["LOSS_TYPE"]= "sum_weigh_linear"
#p["LOSS_TYPE"]= "avg_xentropy"

#p["AUGMENTATION"]= {}

p["AUGMENTATION"]= {
    "random_shift": 40.0,
}

#p["REDUCED_CLASSES"]= [0]

p["SPEAKER_LEFT"]= 11
p["COLLECT_CONFUSION"]= True
p["TAU_ACCUMULATOR"]= 5.0

p["HIDDEN_NOISE"]= 0.002

"""
p["REC_NEURONS"]= [("output","avgInback"),("accumulator","V")]
p["REC_NEURONS_EPOCH_TRIAL"]= [(0,25),(0,26),(0,27),(0,28),(0,29),(0,30),(0,31),
                               (9,25),(9,26),(9,27),(9,28),(9,29),(9,30),(9,31),
]

p["REC_SPIKES_EPOCH_TRIAL"]= [(0,25),(0,26),(0,27),(0,28),(0,29),(0,30),(0,31),
                              (9,25),(9,26),(9,27),(9,28),(9,29),(9,30),(9,31),
]
p["REC_SPIKES"]= ["input","hidden"]

p["W_OUTPUT_EPOCH_TRIAL"]= [(0,25),(0,26),(0,27),(0,28),(0,29),(0,30),(0,31),
                              (9,25),(9,26),(9,27),(9,28),(9,29),(9,30),(9,31),
]
"""

if p["DEBUG"]:
    p["REC_SPIKES"]= ["input", "hidden"]
    #p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I")]
    #p["REC_SYNAPSES"]= [("hid_to_out", "w")]

"""
p["REC_SPIKES"]= ["input","hidden"]
p["REC_SPIKES_EPOCH_TRIAL"]= [(0,0), (100,0), (100,1), (100,2), (100,28), (100,29),
                              (799,0), (799,1), (799,2), (799,28), (799,29),]

p["REC_NEURONS"]= [("output","V")]
p["REC_NEURONS_EPOCH_TRIAL"]= [(0,0), (100,0), (100,1), (100,2), (100,28), (100,29),
                              (799,0), (799,1), (799,2), (799,28), (799,29),]

p["W_OUTPUT_EPOCH_TRIAL"]= [(0,0), (100,0), (799,0)]
"""

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as file:
    json.dump(p, file)
    
mn= SHD_model(p)

interrupt= False
try:
    spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
    #spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.cross_validate_SHD(p)
except KeyboardInterrupt:
    interrupt= True

found= False
i= -1
while not found:
    i=i+1
    jname= p["NAME"]+'.'+str(i)+'.json'
    found= not exists(jname)

jfile= open(jname,'w')
json.dump(p,jfile)
rname= p["NAME"]+'.'+str(i)+'.summary.txt'
sumfile= open(rname,'w')
if interrupt:
    label= 'Cancelled: '
else:
    label= 'Complete: '
    
sumfile.write("{}Training correct: {}, Valuation correct: {}".format(label,correct,correct_eval))

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
