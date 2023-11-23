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

p["WRITE_TO_DISK"]= True
p["N_EPOCH"] = 50
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["N_INPUT_DELAY"]= 10
p["NUM_HIDDEN"]= 256
p["INPUT_DELAY"]= 40.0
p["RECURRENT"]= False
p["N_BATCH"]= 32
p["INPUT_HIDDEN_MEAN"]= 0.002
p["INPUT_HIDDEN_STD"]= 0.02
p["ETA"]= 1e-3
p["LBD_UPPER"]= p0["LBD_UPPER"]*0.05
p["LBD_LOWER"]= p0["LBD_LOWER"]*0.05
p["BALANCE_TRAIN_CLASSES"]= True
p["REWIRE_SILENT"]= True
p["REWIRE_LIFT"]= 0.003
p["HIDDEN_NEURON_TYPE"]= "hetLIF"

#p["REC_NEURONS"] = [("hidden0","lambda_V"),("hidden0","lambda_I"), ("output","lambda_V"), ("output","lambda_I")]
#p["REC_NEURONS_EPOCH_TRIAL"] = [ [0,1], [0,2], [0,3], [0,4] ]
#p["REC_SPIKES"]= ["input","hidden0"]
#p["REC_SPIKES_EPOCH_TRIAL"] = [ [0,0], [0,1], [0,2] ]

p["COLLECT_CONFUSION"]= True
print(p)
with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'),'w') as f:
    json.dump(p, f)

#exit(1)

mn= SHD_model(p)
#res= mn.cross_validate_SHD(p)
res= mn.train(p)


print(res)

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
lambda_V= res[2]["lambda_Vhidden0"]
lambda_I= res[2]["lambda_Ihidden0"]
lambda_V_o= res[2]["lambda_Voutput"]
lambda_I_o= res[2]["lambda_Ioutput"]

for k in range(1):
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
"""

#plt.figure()
#plt.scatter(res[0]['hidden0'],res[1]['hidden0'],s=0.1)
#plt.figure()
#plt.scatter(res[0]['input'],res[1]['input'],s=0.1)
#plt.show()

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_summary.json'),'w') as f:
    json.dump(res, f)


