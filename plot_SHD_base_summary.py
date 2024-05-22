import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from utils import incr, print_diff, diff_keys, subdict

if len(sys.argv) != 1:
    print("usage: python plot_SHD_summary_v4.py ")
    exit(1)

"""
This script plots the performances on teh base SHD model using data from 
scan_SHD_base_xval
scan_SHD_base_traintest
published on figshare:
"""
    
xbname= "scan_SHD_base_xval/SHD_xval"
ttbname= "scan_SHD_base_traintest/SHD_tt"

N_spk = 10
N_cond = 8
N_rep = 8
N_E= 300
ids = [17, 21, 42, 62, 90, 94, 139, 143] # from plot_scan_results_xval on scan_JUWELS_45
con= [[ "sum", "sum exp", "first spike", "max"], ["sum", "sum exp", "first spike",  "max"]] 

y = np.zeros((2, 4),dtype=int)
x = np.zeros((2, 4, 4))
z = np.zeros((2, 4, 4))
# do two separate figures ffwd and recurrent
for r in range(2):
    for j in range(0,N_cond//2):
        # do xval first
        best = []
        btr = []
        btst = []
        id = ids[j*2+r]
        with open(xbname+"_"+str(id).zfill(4)+"_results.txt", "r") as f:
            d = np.loadtxt(f)
        for spk in range(N_spk):
            pos = np.argmax(d[spk*N_E:(spk+1)*N_E,3]) # position of best training
            best.append(pos)
            btr.append(d[spk*N_E+pos,1])
            btst.append(d[spk*N_E+pos,3])
        x[r,j,0] = np.mean(btr)
        x[r,j,1] = np.mean(btst)
        z[r,j,0] = np.std(btr)
        z[r,j,1] = np.std(btst)
        
        train = []
        test = []
        for k in range(N_rep):
            id = (j*2+r)*N_rep+k
            with open(ttbname+"_"+str(id).zfill(4)+"_results.txt", "r") as f:
                d= np.loadtxt(f)
                train.append(d[:,1])
                test.append(d[:,3])
        train = np.vstack(train)
        test = np.vstack(test)
        mtr = np.mean(train,axis=0)
        sigtr = np.std(train,axis=0)
        mtst = np.mean(test,axis=0)
        sigtst = np.std(test,axis=0)
        y[r,j] = int(np.argmax(mtr[:N_E]))
        x[r,j,2] = mtr[y[r,j]]
        x[r,j,3] = mtst[y[r,j]]
        z[r,j,2] = sigtr[y[r,j]]
        z[r,j,3] = sigtst[y[r,j]]
        
    
    plt.figure(figsize=[ 4, 3 ])
    #plt.grid(visible=True, which= 'major', axis= 'y', color='k', linestyle=':', linewidth= 0.5)
    for m in range(4):
        plt.bar(np.arange(4)+0.2*m, x[r,:,m],width=0.18,zorder=5)
    for m in range(4):
        plt.errorbar(np.arange(4)+0.2*m, x[r,:,m], z[r,:,m],ls='none',color='k',zorder= 10)
    print(x[r,:,:])
    print(z[r,:,:])
    plt.gca().set_yticks(np.arange(0.1,1.0,0.1), minor=True)
    plt.gca().set_yticks(np.arange(0,1.1,0.5))
    plt.gca().tick_params(which='minor', length=0)
    plt.gca().grid(axis="y")
    plt.gca().grid(which='minor', alpha=0.3)
    plt.ylim([ 0, 1.1])
    plt.xticks(np.arange(4)+0.3)
    plt.gca().set_xticklabels(con[r]) #, rotation= 60)
    plt.ylabel("fraction correct")
    #plt.legend(["training xval", "validation xval", "training", "test"],loc="upper right")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("SHD_all_overview_"+str(r)+".pdf")
plt.show()

# sort out the used parameter values (this is similar code to get_best_values_paras.py)

def blank_irrelevant(p):
    p["NAME"]= "None"
    p["OUT_DIR"]= "None"
    p["TRAIN_DATA_SEED"]= "None"
    p["TEST_DATA_SEED"]= "None"
    p["MODEL_SEED"]= "None"
    return p

diffkeys= set()
for i in range(len(ids)):
    with open(xbname+"_"+str(ids[i]).zfill(4)+".json","r") as f:
        p= json.load(f)
    p= blank_irrelevant(p)
    for j in range(i+1,len(ids)):
        with open(xbname+"_"+str(ids[j]).zfill(4)+".json","r") as f:
            p1= json.load(f)
        p1= blank_irrelevant(p1)
        d= diff_keys(p, p1)
        diffkeys= diffkeys.union(d)

for i in range(len(ids)):
    with open(xbname+"_"+str(ids[i]).zfill(4)+".json","r") as f:
        p= json.load(f)
    pnew= subdict(p,diffkeys)
    fname= xbname+"_"+str(ids[i]).zfill(4)+"_unique_p.txt"
    print(f"writing unique parameters to {fname}")
    with open(fname,"w") as f:
        for name, value in pnew.items():
            f.write(f"{name}: {value} \n")
        f.close()
        
# note the common parameter values shared by all
with open(xbname+"_"+str(ids[0]).zfill(4)+".json","r") as f:
    p= json.load(f)

fname= xbname+"_best_common_p.txt"
print(f"writing common parameters to {fname}")
with open(fname, "w") as f:
    commkeys= p.keys() - diffkeys
    for k in commkeys:
        f.write(f"{k}: {p[k]} \n")
    f.close()
