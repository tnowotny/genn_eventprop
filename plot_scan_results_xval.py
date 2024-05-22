import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from utils import gridlines, remap, average_out, optimise

title = False
axis_labels = False
mark_opt = False
show_rank = True
N_rank = 10

if len(sys.argv) != 3:
    print("Usage: python plot_scan_results.py <basename> <settings.json>")
    exit(1)

basename= sys.argv[1]

with open(sys.argv[2],"r") as f:
    para= json.load(f)

    
N_S= para["epoch"] # show results at this epoch

N_avg= 5
N_spk= 10
s= np.product(para["split"])

results = [ [] for i in range(11) ] # 11 types of results
res_names = [ "id", "total number epochs", "final train correct", "final train loss", "final valid correct", "final valid loss", "epoch of best train", "best train correct", "train loss at best train", "valid correct at best train", "valid loss at best train" ]
res_col = 11 # includes one column for id
opt_col = 9 # that is best validation performance

results = np.zeros((11,s))
for i in range(s):
    fail = False
    id = str(i)
    if "pad_ID" in para:
        id = id.zfill(4)
    fname= basename+"_"+id+".json"
    try:
        with open(fname,"r") as f:
            p= json.load(f)
    except:
        print(f"error trying to load {fname}")
    else:
        N_E= p["N_EPOCH"] # total number of epochs for each speaker
        tN_S= min(p["N_EPOCH"], N_S)
        idx= np.outer(np.arange(tN_S-N_avg,N_spk*N_E,N_E),np.ones(N_avg))+np.outer(np.ones(N_spk),np.arange(N_avg))
        idx= np.array(idx.flatten(), dtype= int)
        fname= basename+"_"+id+"_results.txt"
        results[0,i]= i
        try:
            with open(fname, "r") as f:
                d= np.loadtxt(f)
        except:
            print("error trying to load {}".format(fname))
        else:
            try:
                results[1,i]= len(d[:,1])
                for j in range(1,5):
                    results[j+1,i] = np.mean(d[idx,j])
                    # process best positions
                best = [ [] for i in range(5)]
                for spk in range(N_spk):
                    pos = np.argmax(d[spk*N_E:(spk+1)*N_E,1]) # position of best training
                    best[0].append(pos)
                    for j in range(1,5):
                        best[j].append(d[pos,j])
                for j in range(5):
                    results[j+6,i] = np.mean(best[j])
            except:
                print(f"error trying to average for {fname}")

results= np.array(results)
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

if show_rank:
    rnk = np.argsort(-results[opt_col])
    
print(f"Original IDs of best (averaged if average is on!): {list(results[0,opt].astype(int))}")
# decide on x/y split
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
    plt.imshow(results[i,:].reshape((ht,wd)),vmin= 0.5,#vmin= np.median(results[i,:]),
               interpolation='none',cmap='jet')
    print(f"i: {i}, {res_names[i]}, max: {np.amax(results[i,:])}")
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    if "explicit_lines" in para:
        extra_lines= para["explicit_lines"]
    else:
        extra_Lines = None
    gridlines(plt.gca(),hti, wdi, split, extra_lines=extra_lines)

    if title:
        plt.title(res_names[i])
    if axis_labels:
        plt.ylabel(ylbl)
        plt.xlabel(xlbl)
    if mark_opt:
        plt.scatter( np.mod(opt,wd), np.asarray(opt,dtype=int) // wd, marker='x', color= 'w')
    if show_rank:
        for j in range(N_rank):
            plt.text(np.mod(rnk[j],wd)-0.2, (rnk[j] // wd)+0.25, str(j),color="w",fontsize= 8)
    print(f"{res_names[i]} top values: {results[i,rnk[:8]]}, mean= {np.mean(results[i,rnk[:8]])}, std= {np.std(results[i,rnk[:8]])}")
    plt.tight_layout()
    plt.savefig(basename+"_"+res_names[i].replace(" ","_")+".pdf")

plt.show()
