import json
import numpy as np
import sys

"""
ad-hoc script to identify the final test performance for the observed best-performing
cross-validation (SHD) or train-test (SSC) runs
"""

if len(sys.argv) != 2 or (sys.argv[1] != "SHD" and sys.argv[1] != "SSC"):
    print(f"usage: {sys.argv[0]} <which> where which == \"SHD\" or \"SSC\"")
    exit(1)
    
which = sys.argv[1]

if which == "SHD":
    # from plot_scan_results_xval.py on scan_SHD_final_xval/SHD_xval
    # [1406]: eval 0.9200438116100768

    # index of best cross-validation result
    b= 1406*2

    basename = "scan_SHD_final_xval/SHD_xval"
    bname = "scan_SHD_final_traintest/SHD_tt"
    nn = 2560
    extension= "best"

if which == "SSC":
    # from plot_scan_results_traineval.py scan_SSC_final/SSC scan_SSC_final/split.json 
    # [792]: eval 0.7454162909528104
    # index of best train-eval run
    b = 792*2
    basename = "scan_SSC_final/SSC"
    bname = "scan_SSC_final_repeats/SSC"
    nn = 1920
    extension= "test_results"
    
fname = basename+"_"+str(b)+".json"
with open(fname,"r") as f:
    p= json.load(f)

def blank_irrelevant(p):
    p["NAME"] = ""
    p["OUT_DIR"] = ""
    p["TRAIN_DATA_SEED"] = ""
    p["TEST_DATA_SEED"] = ""
    p["MODEL_SEED"] = ""
    p["N_EPOCH"] = ""
    p["CHECKPOINT_BEST"] = ""
    p["ORIG_NAME"] = ""
    p["SPEAKER_LEFT"] = ""
    return p

p = blank_irrelevant(p)
    
x = []
if which == "SSC":
    for i in range(2):
        fname = basename+"_"+str(b+i)+"_test_results.txt"
        d = np.loadtxt(fname)
        x.append(d[3])
    
for i in range(nn):
    fname = bname+"_"+str(i).zfill(4)+".json"
    try:
        with open(fname,"r") as f:
            p0= json.load(f)
    except:
        pass
    else:
        p0 = blank_irrelevant(p0)
        if p == p0:
            print(f"found match {fname}!")
            fname = bname+"_"+str(i).zfill(4)+"_best.txt"
            d= np.loadtxt(fname)
            x.append(d[3])

print(f"Mean accuracy on test for best xval: {np.mean(x)}, std: {np.std(x)}, N={len(x)}")


