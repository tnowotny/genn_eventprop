from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

"""
Take an older json dump of SHD training parameters and update so that it is the best guess 
what should be used with the latest version of simulator_SHD.
This includes assuming default values for new parameters and omitting parameters that are no
longer in use.
"""

if len(sys.argv) != 3:
    print("usage: python update_SHD_json XXX.json NEWNAME")
    exit(1)

with open(sys.argv[1],"r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    if name in p:  # only update if this parameter is still in use
        p[name]= value

p["NAME"]= sys.argv[2]

jname= p["NAME"]+".json"
jfile= open(jname,'w')
json.dump(p,jfile)

