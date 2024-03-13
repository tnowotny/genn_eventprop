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

#p["N_EPOCH"] = 100

#p["AUGMENTATION"]= {}

#p["CHECKPOINT_BEST"] = "validation"
#p["SPEAKER_LEFT"] = list(range(10))
#p["TAU_SYN"] = 10.0
#p["TRAIN_TAU"] = False
#p["N_BATCH"] = 256
#p["ETA"] = 1e-3
#p["ETA"] *= 0.1
p["LR_EASE_IN_FACTOR"] = 1.05
#p["N_HID_LAYER"] = 2

print(p)
fname= p["NAME"]+".json"
with open(os.path.join(p["OUT_DIR"], fname),"w") as f:
    json.dump(p, f)

mn= SHD_model(p)
#res= mn.cross_validate_SHD(p)
#res= mn.train_test(p)
res= mn.train(p)

np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_res"), res)
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

"""
tau_m= res[2]["tau_mhidden0"]
lambda_V= res[2]["lambda_Vhidden0"]
lambda_I= res[2]["lambda_Ihidden0"]
lambda_V_o= res[2]["lambda_Voutput"]
lambda_I_o= res[2]["lambda_Ioutput"]

for k in range(5):
    fig, axes= plt.subplots(8,4)
    for i in range(8):
        for j in range(4):
            n= i*4+j
            axes[i,j].plot(tau_m[1000*k:1000*(k+1),n])

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

"""
