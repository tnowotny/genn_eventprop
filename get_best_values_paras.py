import json
import numpy as np
from utils import incr, print_diff, diff_keys, subdict
import sys


def blank_irrelevant(p):
    p["NAME"]= "None"
    p["OUT_DIR"]= "None"
    p["TRAIN_DATA_SEED"]= "None"
    p["TEST_DATA_SEED"]= "None"
    p["MODEL_SEED"]= "None"
    return p

if len(sys.argv) != 2:
    print("usage: get_best_values_paras.py <output base name>")
unique_name= sys.argv[1]

loss_types= ["sum", "sum_weigh_exp", "first_spike_exp", "max"]
recur_types= ["feedforward", "recurrent"]
augment_types= ["None", "random_shift"]

# cross-validation scans:
sources= [{
    "path": "scan_JUWELS_21/J21_scan_",
    "rngs": [ 4, 2, 2, 2, 2, 2, 2, 2, 2 ],
    "loss": 0,
    "recur": 2,
    "augment": 7,
},
{
    "path": "scan_JUWELS_23/J23_scan_",
    "rngs": [ 4, 2, 2, 2, 2, 2, 2, 2 ],
    "loss": 0,
    "recur": 2,
    "augment": 6,
},
{
    "path": "scan_JADE_29/J29_scan_",
    "rngs": [ 6, 2, 2 ],
    "loss": "max",
    "recur": "feedforward",
    "augment": "random_shift",
}
]

# train_test final runs:
tt_sources= ["scan_JADE_26/J26_scan_",
             "scan_JADE_27/J27_scan_",
             "scan_JADE_28/J28_scan_",
             "scan_JUWELS_24/J24_scan_",
             "scan_JUWELS_25/J25_scan_",
             "scan_JUWELS_26/J26_scan_",
             ]

N_E= 300
N_S= 300
idx= np.outer(np.arange(N_S-10,10*N_E,N_E),np.ones(10))+np.outer(np.ones(10),np.arange(10))
idx= np.array(idx.flatten(), dtype= int)

best= []
for a in range(2):
    for l in range(4):
        for r in range(2):
            csrc= []
            mx_train= []
            mx_train_std= []
            mx_eva= []
            mx_eva_std= []
            mx_tid= []
            for src in sources:
                print(src)
                if (((type(src["loss"]) == int) or (src["loss"] == loss_types[l])) and
                    ((type(src["recur"]) == int) or (src["recur"]== recur_types[r])) and
                    ((type(src["augment"]) == int) or (src["augment"] == augment_types[a]))):
                    # need to consider this source
                    fx= []
                    fxv= []
                    if type(src["loss"]) == int:
                        fx.append(src["loss"])
                        fxv.append(l)
                    if type(src["recur"]) == int:
                        fx.append(src["recur"])
                        fxv.append(r)
                    if type(src["augment"]) == int:
                        fx.append(src["augment"])
                        fxv.append(a)
                    # fix adam optimizer to 0.9/0.999
                    fx.append(6)
                    fxv.append(0)
                    # fix pdrop_input to 0
                    fx.append(3)
                    fxv.append(0)
                    # fix hidden noise to 0
                    fx.append(5)
                    fxv.append(0)
                    
                    i= incr(src["rngs"], fix= fx, fix_v= fxv)
                    train= []
                    train_std= []
                    eva= []
                    eva_std= []
                    tid= []
                    while i.val() is not None:
                        v= i.val()
                        #print(i.x)
                        #print(v)
                        fname= src["path"]+str(v)+"_results.txt"
                        #print(fname)
                        try:
                            with open(fname, "r") as f:
                                d= np.loadtxt(f)
                        except:
                            print("error trying to load {}".format(fname))
                        else:
                            tid.append(v)
                            train.append(np.mean(d[idx,1]))
                            train_std.append(np.std(d[idx,1]))
                            eva.append(np.mean(d[idx,3]))
                            eva_std.append(np.std(d[idx,3]))
                            if l == 3 and a == 0:
                                print(f"max train: {np.mean(d[idx,1])}, val: {np.mean(d[idx,3])}")
                        i.next()
                    # average the two neighbouring results that only differ in the seeds
                    train= np.mean(np.array(train).reshape(len(train)//2,2),axis=1)
                    train_std= np.mean(np.array(train_std).reshape(len(train_std)//2,2),axis=1)
                    eva= np.mean(np.array(eva).reshape(len(eva)//2,2),axis=1)
                    eva_std= np.mean(np.array(eva_std).reshape(len(eva_std)//2,2),axis=1)
                    pos= np.argmax(np.array(eva))
                    csrc.append(src["path"])
                    mx_train.append(train[pos])
                    mx_train_std.append(train_std[pos])
                    mx_eva.append(eva[pos])
                    mx_eva_std.append(eva_std[pos])
                    mx_tid.append(tid[pos])
            pos= np.argmax(np.array(mx_eva))
            best.append({
                "path": csrc[pos],
                "id": int(mx_tid[pos]),
                "perf": [mx_train[pos], mx_train_std[pos], mx_eva[pos], mx_eva_std[pos]]
                })


print(best)
# ok, we have the best performers based on cross-validation performance, let's match the train_test runs

for b in best:
    fname= b["path"]+str(b["id"])+".json"
    with open(fname,"r") as f:
        p= json.load(f)
    p= blank_irrelevant(p)    
    found= False
    for src in tt_sources:
        if not found:
            done= False
            i= 0
            while not done:
                fname1= src+str(i)+".json"
                print(fname1)
                try:
                    with open(fname1,"r") as f:
                        p1= json.load(f)
                except:
                    done= True
                p1= blank_irrelevant(p1)
                #print_diff(p,p1)
                if p == p1:
                    found= True
                    done= True
                else:
                    i+= 1 
                if found:
                    b["id2"]= i
                    b["path2"]= src
                    fname1= fname1[:-5]+"_results.txt"
                    try:
                        with open(fname1, "r") as f:
                            d= np.loadtxt(f)
                    except:
                        print("error trying to load {}".format(fname1))
                    else:
                        b["perf"].append(np.mean(d[idx,1]))
                        b["perf"].append(np.std(d[idx,1]))
                        b["perf"].append(np.mean(d[idx,3]))
                        b["perf"].append(np.std(d[idx,3]))

print(best)
for x in best:
    print(len(x["perf"]))

# sort out what the final choices for parameters were
diffkeys= set()
for i in range(len(best)):
    with open(best[i]["path"]+str(best[i]["id"])+".json","r") as f:
        p= json.load(f)
    p= blank_irrelevant(p)
    for j in range(i+1,len(best)):
        with open(best[j]["path"]+str(best[j]["id"])+".json","r") as f:
            p1= json.load(f)
        p1= blank_irrelevant(p1)
        d= diff_keys(p, p1)
        diffkeys= diffkeys.union(d)

for i in range(len(best)):
    with open(best[i]["path"]+str(best[i]["id"])+".json","r") as f:
        p= json.load(f)
    pnew= subdict(p,diffkeys)
    best[i]["unique_p"]= pnew
        
with open(unique_name+"best_summary.json","w") as f:
    json.dump(best,f)

with open(unique_name+"best_unique_p.txt", "w") as f:
    idx= [ 0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15 ]
    nbest= [ best[i] for i in idx ]
    best= nbest

    for k in diffkeys:
        f.write(f"{k} & ")
        for b in best:
            f.write(f" {b['unique_p'][k]} &")
        f.write("\n")
    f.close()

# note the common parameter values shared by all
with open(best[0]["path"]+str(best[0]["id"])+".json","r") as f:
    p= json.load(f)

with open(unique_name+"best_common_p.txt", "w") as f:
    commkeys= p.keys() - diffkeys
    for k in commkeys:
        f.write(f"{k} {p[k]} \n")
    f.close()
    
"""
outputs: for each combination of recurrent, augmented, loss type:
- the dirname and the ID of the best performer
- the best training and validation mean
- the best training and validation std
- the dictionary of bespoke parameters

For the total set:
- the dictionary of common parameters
"""
