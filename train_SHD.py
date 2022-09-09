from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json

p["TRIAL_MS"]= 1400
p["DATASET"]= "SHD"
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["MODEL_SEED"]= 135
p["NAME"]= "test101"
p["NUM_HIDDEN"]= 256
p["N_MAX_SPIKE"]= 1500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.1
p["PDROP_HIDDEN"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["DEBUG"]= False
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 100
p["N_BATCH"]= 32
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 7900 #20*p["N_BATCH"] #7756 
p["N_VALIDATE"]= 512 # 256 # p["N_BATCH"] 
p["ETA"]= 2e-3 #1e-4 # 5e-3
p["SHUFFLE"]= True
p["INPUT_HIDDEN_MEAN"]= 0.02 # 0.02
p["INPUT_HIDDEN_STD"]= 0.01 # 0.01
p["HIDDEN_OUTPUT_MEAN"]= 0.0 # 0.0
p["HIDDEN_OUTPUT_STD"]= 0.3 #0.3
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
p["TAU_MEM"] = 20.0 #20
p["TAU_SYN"] = 5.0 #5
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 2e-9 # 2e-9 # 2e-8 # 2e-14 (since removal of N_Batch), 5e-12 keep in mind that the term is applied to all contributing spikes ...
p["LBD_LOWER"]= 2e-9 #2e-8
p["NU_UPPER"]= 14
p["NU_LOWER"]= 5
p["RHO_UPPER"]= 10000.0
p["GLB_UPPER"]= 1e-8
p["ETA_DECAY"]= 1.0      
p["ETA_FIDDELING"]= False
p["ETA_REDUCE"]= 0.5
p["ETA_REDUCE_PERIOD"]= 50
p["TIMING"]= True
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["LOSS_TYPE"]= "sum"
p["EVALUATION"]= "speaker"

p["RECURRENT"]= True
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02 # 0.02

p["REWIRE_SILENT"]= True
p["AVG_SNSUM"]= True

#p["AUGMENTATION"]= {}

p["AUGMENTATION"]= {
    "random_shift": 20.0,
    "random_dilate": (0.9, 1.1),
    "ID_jitter": 5.0
}

#p["REDUCED_CLASSES"]= [0]

"""
p["REC_NEURONS"]= [("output","V"),("output","sum_V"),("output","lambda_V"),("output","lambda_I"),
                   ("hidden","lambda_V")]
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
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
if p["DEBUG"]:
    plt.figure()
    plt.scatter(spike_t["input"],700-spike_ID["input"],s=0.5)
    plt.figure()
    plt.scatter(spike_t["hidden"],spike_ID["hidden"],s=0.5)
    #tmax= np.max(spike_t["hidden"])
    #t= np.arange(0,tmax,p["TRIAL_MS"])
    #v= np.vstack([ np.zeros(1,lent(t)), np.ones(1,len(t))])
    #t= np.reshape(t,(1,len(t)))
    #t= np.vstack([ t, t ])
    #plt.plot(t,v, lw=0.1)
    #plt.figure()
    #plt.plot(rec_vars_n["Voutput"])
    #plt.figure()
    #plt.plot(rec_vars_n["lambda_Voutput"])
    #plt.figure()
    #plt.plot(rec_vars_n["lambda_Ioutput"])
    #print(mn.output.extra_global_params["label"].view[0:20*p["N_BATCH"]:p["N_BATCH"]])
    #plt.figure()
    #h, x= np.histogram(spike_ID["hidden"],p["NUM_HIDDEN"])
    #h= np.sort(h)
    #plt.bar(np.arange(len(h)),np.log(h+2))
#mn.hid_to_out.pull_var_from_device("w")
#plt.figure()
#plt.hist(mn.hid_to_out.vars["w"].view[:],100)
plt.show()
