from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_JUWELS_41e/"
# learning rate schedule depending on exponential moving average of performance
p["MIN_EPOCH_ETA_FIXED"]= 50
p["EMA_ALPHA2"]= 0.85
p["EMA_ALPHA1"]= 0.8
p["ETA_FAC"]= 0.5
p["TRIAL_MS"]= 1400.0
p["NU_UPPER"]= 14
p["N_HID_LAYER"]= 1
p["BALANCE_TRAIN_CLASSES"]= False
p["BALANCE_EVAL_CLASSES"]= False
p["RESCALE_T"]= 1.0
p["RESCALE_X"]= 1.0
p["DATA_SET"]= "SHD"
p["N_BATCH"]= 256
p["N_TRAIN"]= 7644 
p["N_VALIDATE"]= 512 
p["REWIRE_SILENT"]= True
p["REWIRE_LIFT"]= 0.002
p["TRAIN_TAU"]= False
p["COLLECT_CONFUSION"]= True
p["N_EPOCH"] = 300
p["AUGMENTATION"]= {
}
p["EMA_WHICH"] = "validation"
p["CHECKPOINT_BEST"] = "neither"
p["HIDDEN_OUTPUT_MEAN"]= 0.0
p["HIDDEN_OUTPUT_STD"]= 0.03
p["ETA"] = 5e-3
p["LOSS_TYPE"] = "sum"

lbd_fac= [ 0.1, 0.5, 1.0, 5.0, 10.0 ]
recurrent= [ False, True ]
taum= [ 1.0, 2.0 ]
taus= [ 2.0, 4.0 ]
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["SPEAKER_LEFT"]= list(range(10))

for j in range(5):
    for k in range(2):
        for l in range(2):
            for m in range(2):
                id= ((j*2+k)*2+l)*2+m
                print(id)
                p["LBD_UPPER"]=lbd_fac[j]*1e-12
                p["LBD_LOWER"]=lbd_fac[j]*1e-12
                p["RECURRENT"] = recurrent[k]
                p["TAU_MEM"]= p0["TAU_MEM"]*taum[l]
                p["TAU_MEM_OUTPUT"]= p0["TAU_MEM_OUTPUT"]*taum[l]
                p["TAU_SYN"]= p0["TAU_SYN"]*taus[m]
                p["NAME"] = "J41e_scan_"+str(id)
                with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                    json.dump(p, f)
