import numpy as np
import matplotlib.pyplot as plt
import json
import sys

if len(sys.argv) < 2:
    print("usage: python plot_SHD_base_curves.py <basname> (of the scan with train-test results)")
    exit(1)

"""
Script to plot learning curves for the base SHD models.
Works with the data
scan_SHD_base_traintest
published on figshare
"""
    
bname= sys.argv[1]

ptrain= []
ltrain= []
ptest= []
ltest= []

N_rep= 8
N_cond = 8
N_epoch= 300

con= [ "sum ffwd","sum recur","sum exp ffwd", "sum exp recur", "first spike ffwd", "first spike recur", "max ffwd", "max recur" ] 

lng= 0.1
hw = lng*N_epoch/5
hl = 1./N_epoch*1.5*hw

fig, ax= plt.subplots(2,4,sharex=True, sharey=True, figsize= (8,3.5))
for i in range(N_cond):
    train = []
    test = []
    for j in range(N_rep):
        id = i*N_rep+j
        with open(bname+"_"+str(id).zfill(4)+"_results.txt", "r") as f:
            d= np.loadtxt(f)
        train.append(d[:,1])
        test.append(d[:,3])
    train = np.vstack(train)
    test = np.vstack(test)
    mtr = 1.0-np.mean(train,axis=0)
    sigtr = np.std(train,axis=0)
    mtst = 1.0-np.mean(test,axis=0)
    sigtst = np.std(test,axis=0)
    mn1 = np.argmin(mtr[:N_epoch])
    mn2 = np.argmin(mtst[:N_epoch])
    x, y = divmod(i,2)
    print(x,y)
    ax[y,x].arrow(mn1,mtr[mn1]+lng,0,-lng,width=hw,head_length=hl,length_includes_head=True,color='k',zorder=10)
    ax[y,x].arrow(mn2,mtst[mn2]+lng,0.0,-lng,width=hw,head_length=hl,length_includes_head=True,color='k',zorder=10)
    ax[y,x].plot(range(N_epoch),mtr[:N_epoch],zorder=5)
    ax[y,x].fill_between(range(N_epoch),mtr[:N_epoch]-sigtr[:N_epoch],mtr[:N_epoch]+sigtr[:N_epoch],alpha=0.4,zorder=0)
    ax[y,x].plot(range(N_epoch),mtst[:N_epoch],zorder=5)
    ax[y,x].fill_between(range(N_epoch),mtst[:N_epoch]-sigtst[:N_epoch],mtst[:N_epoch]+sigtst[:N_epoch],alpha=0.4,zorder=0)
    ax[y,x].set_yticks(np.arange(0,1.1,0.5))
    ax[y,x].set_yticks(np.arange(0.1,1.0,0.1), minor=True)
    ax[y,x].tick_params(which='minor', length=0)
    ax[y,x].grid(axis="y")
    ax[y,x].grid(which='minor', alpha=0.3)
    ax[y,x].set_xticks([ 1, 100, 200, 300 ])
    ax[y,x].set_ylim([0,1])
    ax[y,x].spines['top'].set_visible(False)
    ax[y,x].spines['right'].set_visible(False)
    ax[y,x].set_title(con[i])
fig.tight_layout()
fig.savefig("SHD_curves.pdf")
plt.show()
