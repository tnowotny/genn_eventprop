import json
import sys

p= [ {}, {} ]
for i in range(2):
    fname= sys.argv[i+1]
    with open(fname,"r") as f:
        p[i]= json.load(f)

for lbl, val in p[0].items():
    if lbl not in p[1]:
        print(f"{lbl} not in {sys.argv[2]}")
    else:
        if p[1][lbl] != val:
            print(f"{lbl}: {val} - {p[1][lbl]}")
        
