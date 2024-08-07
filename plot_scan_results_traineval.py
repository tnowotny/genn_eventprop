import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from utils import gridlines, remap, average_out, optimise, load_train_test

title= True
axis_labels= True
mark_opt = False
show_rank = True
N_rank = 10

#RESERVE = "scan_JUWELS_40b2/J40b_scan"
RESERVE = None

if len(sys.argv) != 3:
    print("Usage: python plot_scan_results.py <basename> <settings.json>")
    exit(1)

basename= sys.argv[1]

with open(sys.argv[2],"r") as f:
    para= json.load(f)
    
N_avg= 5
s= np.product(para["split"])

res_names = [ "id", "total number epochs", "final train correct", "final train loss", "final valid correct", "final valid loss", "epoch of best valid", "train correct at best valid", "train loss at best valid", "best valid correct", "valid loss at best valid", "time per epoch secs" ]
opt_col = 9 # that is best validation performance
if "pad_ID" in para:
    pad_ID = para["pad_ID"]
else:
    pad_ID = None
results= load_train_test(basename,s,N_avg,"best",reserve=RESERVE,pad_ID=pad_ID)

print(results.shape)
   

if "avg" in para:
    results, s, split, names = average_out(results, s, para["split"], para["avg"], para["names"]) 
else:
    split = para["split"]
    names = para["names"]
    
mx= np.max(results[7])
p75 = np.percentile(results[7],50)
nmax = list(np.where(results[7] == mx)[0])
mx_e = np.max(results[opt_col])
p75_e = np.percentile(results[opt_col],50)
nmax_e = list(np.where(results[opt_col] == mx_e)[0])
print(f"{nmax}: train {mx}")
print(f"{nmax_e}: eval {mx_e}")

if "remap" in para:
    results, split = remap(results, split, para["remap"])
else:
    para["remap"]= list(range(len(split)))

if "list" in para and "optimise" in para:
    opt = optimise(results[opt_col], para["list"], para["optimise"], split)
    print(f"Original IDs of best (averaged if average is on!): {list(results[0,opt].astype(int))}")


if show_rank:
    rnk = np.argsort(-results[opt_col])    
    
# decide on x/y split
s= np.product(split)
i= 0
ht= 1
while ht < np.sqrt(s):
    ht*= split[i]
    i+= 1

i-= 1
ht= ht // split[i]
hti= i
wd= int(np.product(split[hti:]))
wdi= len(split)-hti

ylbl= ""
xlbl= ""
if len(names) == len(split):
    for i in range(hti):
        ylbl= ylbl+names[para["remap"][i]]+","
    for i in range(wdi):
        xlbl= xlbl+names[para["remap"][hti+i]]+","

for i in range(0,results.shape[0]):
    plt.figure(figsize=(20,5.2))
    plt.imshow(results[i,:].reshape((ht,wd)),vmin=0.4, #vmin= np.median(results[i,:]),
               interpolation='none',cmap='jet')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    if "explicit_lines" in para:
        extra_lines= para["explicit_lines"]
    else:
        extra_lines = None
    gridlines(plt.gca(),hti, wdi, split, extra_lines=extra_lines)

    
    if title:
        plt.title(res_names[i])
    if axis_labels:
        plt.ylabel(ylbl)
        plt.xlabel(xlbl)

    if show_rank:
        for j in range(N_rank):
            plt.text(np.mod(rnk[j],wd)-0.2, (rnk[j] // wd)+0.25, str(j),color="w",fontsize=8)
    if mark_opt:
        plt.scatter( np.mod(opt,wd), np.asarray(opt,dtype=int) // wd, marker='x', color= 'w')

    plt.tight_layout()
    plt.savefig(basename+"_"+res_names[i].replace(" ","_")+".pdf")
      
plt.show()
