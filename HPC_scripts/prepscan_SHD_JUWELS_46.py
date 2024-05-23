from simulator_SHD import p
import os
import json
import numpy as np

ids = [17, 21, 42, 63, 82, 110, 131, 143]  # from plot_scan_results_xval on scan_JUWELS_45

for i in range(len(ids)):
    for j in range(8):
        id= i*8+j
        print(id)
        with open(f"scan_JUWELS_45/J45_scan_{ids[i]}.json","r") as f:
            p0= json.load(f)
            for (name,value) in p0.items():
                p[name]= value
        p["CHECKPOINT_BEST"] = "training"
        p["TRAIN_DATA_SEED"]= p0["TRAIN_DATA_SEED"]+j*11
        p["TEST_DATA_SEED"]= p0["TEST_DATA_SEED"]+j*22
        p["MODEL_SEED"]= p0["MODEL_SEED"]+j*33
        p["OUT_DIR"]= "scan_JUWELS_46/"
        p["NAME"] = "J46_scan_"+str(id)
            
        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
            json.dump(p, f)
