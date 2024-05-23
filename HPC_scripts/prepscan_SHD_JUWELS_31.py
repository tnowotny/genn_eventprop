from simulator_SHD import p
import os
import json
import numpy as np

with open("scan_JUWELS_21/J21_scan_322.json","r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value

p["OUT_DIR"]= "scan_JUWELS_31/"
# learning rate schedule depending on exponential moving average of performance
p["EMA_ALPHA1"]= 0.1
p["EMA_ALPHA2"]= 0.01
p["ETA_FAC"]= 0.5
p["MIN_EPOCH_ETA_FIXED"]= 300

# sample temporal scaling
scale_t= [ 0.4, 0.6, 0.8, 0.9 ]

for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                for m in range(4):
                    id= (((i*4+j)*4+k)*4+l)*4+m
                    print(id)
                    p["RESCALE_T"]= scale_t[i]
                    p["TRIAL_MS"]= 1400*scale_t[i]
                    kk= np.exp(j*np.log(scale_t[i])/3.0)
                    p["TAU_MEM"] = 20.0*kk
                    kk= np.exp(k*np.log(scale_t[i])/3.0)
                    p["TAU_MEM_OUTPUT"] = 20.0*kk
                    kk= np.exp(l*np.log(scale_t[i])/3.0)
                    p["TAU_SYN"] = 5.0*kk
                    kk= np.exp(m*np.log(scale_t[i])/3.0)
                    p["NU_UPPER"]= 14*kk
                    p["NU_LOWER"]= 5*kk
                    p["NAME"]= "J31_scan_"+str(id)
                                    
                    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                        json.dump(p, f)
