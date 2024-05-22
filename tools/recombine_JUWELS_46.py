import numpy as np
import json
import os

basename = "scan_JUWELS_46/J46_scan_"
destination = "scan_SHD_base_traintest"

files = [
    "_best.txt",
    ".json",
    "_results.txt",
    "_summary.json",
    "_w_hidden0_hidden0_best.npy",
    "_w_hidden_output_best.npy",
	"_w_input_hidden_best.npy",
]

for i in range(64):
    newname= f"SHD_tt_{str(i).zfill(4)}"
    for f in files:
        cmd = f"cp {basename}{str(i)}{f} {destination}/{newname}{f}"
        #print(cmd)
        os.system(cmd)
