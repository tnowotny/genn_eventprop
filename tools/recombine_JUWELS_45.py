import numpy as np
import json
import os

basename = "scan_JUWELS_45/J45_scan_"
destination = "scan_SHD_base_xval"

files = [
    ".json",
    "_results.txt",
    "_summary.json",
]

for i in range(160):
    newname= f"SHD_xval_{str(i).zfill(4)}"
    for f in files:
        cmd = f"cp {basename}{str(i)}{f} {destination}/{newname}{f}"
        #print(cmd)
        os.system(cmd)
