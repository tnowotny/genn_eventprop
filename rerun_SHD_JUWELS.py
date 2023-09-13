from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

fname= sys.argv[1]
with open(fname,"r") as f:
    p0= json.load(f)


for k,v in p0.items():
    p[k]= v
    
if len(sys.argv) > 2:
    p["NAME"]+= sys.argv[2]

p["TRAIN_TAUM"]= True

#p["LOSS_TYPE"]= "sum"
#p["TAU_MEM"]= 10.0
#p["TAU_SYN"]*= 1.5
#p["AUGMENTATION"]["blend"]= [0.5, 0.5]  # blend two examples 50:50
#p["AUGMENTATION"]["random_dilate"]= [ 0.95, 1.2 ]
#p["AUGMENTATION"]["ID_jitter"]= 4
#p["AUGMENTATION"]["random_shift"]= 20
#p["N_TRAIN"]*= 3  # do half of training examples as blended in addition to normal
#print(p["AUGMENTATION"])

##p["SPEAKER_LEFT"]= 11
##p["EVALUATION"]= "speaker"
#p["COLLECT_CONFUSION"]= True
p["WRITE_TO_DISK"]= True

p["N_EPOCH"] = 11

#p["AUGMENTATION"]["NORMALISE_SPIKE_NUMBER"]= True
p["BALANCE_TRAIN_CLASSES"]= False
p["BALANCE_EVAL_CLASSES"]= False
p["RESCALE_T"]= 1.0
p["RESCALE_X"]= 1.0
p["TRIAL_MS"]= 1000.0
#p["DT_MS"]= 8.0
#p["AUGMENTATION"]= {}
#p["INPUT_HIDDEN_MEAN"]= 0.1
#p["NU_UPPER"]= 14
#p["NU_LOWER"]= 14

#p["INPUT_HIDDEN_MEAN"]= 0.03
#p["INTPUT_HIDDEN_STD"]= 0.01
#p["HIDDEN_HIDDEN_MEAN"]= 0.0
#p["HIDDEN_HIDDEN_STD"]= 0.02
#p["HIDDEN_OUTPUT_MEAN"]= 0.0
#p["HIDDEN_OUTPUT_STD"]= 0.03
p["N_HID_LAYER"]= 1
p["HIDDEN_HIDDENFWD_MEAN"]= 0.02 # only used when > 1 hidden layer
p["HIDDEN_HIDDENFWD_STD"]= 0.01 # only used when > 1 hidden layer

#p["LBD_UPPER"]= 0 #4e-9
#p["LBD_LOWER"]= 0 #4e-9
#p["N_BATCH"]= 32
#p["ETA"]= 1e-3
#p["RECURRENT"]= True
#p["SHUFFLE"]= False
##p["AVG_SNSUM"]= False

#p["TAU_MEM"]/=10
#p["TAU_MEM_OUTPUT"]/=10
#p["TAU_SYN"]/=10
#p["TRIAL_MS"]=100
p["MIN_EPOCH_ETA_FIXED"]= 50
p["EMA_ALPHA2"]= 0.95
p["EMA_ALPHA1"]= 0.9

p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])


p["REC_NEURONS"] = [("hidden0","ktau_m"),("hidden0","lambda_V"),("hidden0","lambda_I"), ("output","lambda_V"), ("output","lambda_I")]
#p["REC_NEURONS_EPOCH_TRIAL"] = [ [0,1], [0,2], [0,3], [0,4], [0,5], [0,6] ]
p["REC_NEURONS_EPOCH_TRIAL"] = [ [10,1], [10,2], [10,3], [10,4], [10,5], [10,6] ]
#p["REC_SPIKES"]= ["hidden0"]
#p["REC_SPIKES_EPOCH_TRIAL"] = [ [0,1], [0,2], [0,3], [0,4] ]

print(p)
#exit(1)

mn= SHD_model(p)
#res= mn.cross_validate_SHD(p)
res= mn.train_test(p)


#print(res)

"""
dw= res[3]["dwhid_to_out"]
print(dw.shape)
fig, axes= plt.subplots(8,4)
for i in range(8):
    for j in range(4):
        n= i*4+j
        axes[i,j].plot(w[:,n,:])

mdw= np.mean(dw,axis=0)
fig, axes= plt.subplots(8,4)
for i in range(8):
    for j in range(4):
        n= i*8+j
        axes[i,j].plot(dw[n,:])
"""

"""
w= res[3]["whid_to_out"]
print(w.shape)
plt.figure()
plt.plot(w[:,:])
       
plt.figure()
st= res[0]["hidden0"]
sid= res[1]["hidden0"]
plt.scatter(st, sid,s=0.1)
"""

ktau_m= res[2]["ktau_mhidden0"]
lambda_V= res[2]["lambda_Vhidden0"]
lambda_I= res[2]["lambda_Ihidden0"]
lambda_V_o= res[2]["lambda_Voutput"]
lambda_I_o= res[2]["lambda_Ioutput"]

for k in range(5):
    fig, axes= plt.subplots(8,4)
    for i in range(8):
        for j in range(4):
            n= i*4+j
            axes[i,j].plot(ktau_m[1000*k:1000*(k+1),n])

    fig, axes= plt.subplots(8,4)
    for i in range(8):
        for j in range(4):
            n= i*4+j
            axes[i,j].plot(lambda_V[1000*k:1000*(k+1),n])

    fig, axes= plt.subplots(8,4)
    for i in range(8):
        for j in range(4):
            n= i*4+j
            axes[i,j].plot(lambda_I[1000*k:1000*(k+1),n])

    fig, axes= plt.subplots(8,4)
    for i in range(8):
        for j in range(4):
            n= i*4+j
            if n==0:
                print(lambda_V_o[1000*k+270:1000*(k+1),n]-lambda_I_o[1000*k+270:1000*(k+1),n])
            axes[i,j].plot(np.log(np.abs(lambda_V_o[1000*k:1000*(k+1),n]-lambda_I_o[1000*k:1000*(k+1),n])))

plt.show()
#with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_summary.json'),'w') as f:
#    json.dump(res, f)


