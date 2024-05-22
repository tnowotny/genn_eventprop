import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from utils import load_train_test

"""
earlier version of candidate figure for publication - OBSOLETE?
"""

if len(sys.argv) < 4:
    print("Usage: python plot_SHD_paper_augment_hidN.py <basename> <settings.json> <secondary fname part> <basename2> ")
    exit(1)

basename= sys.argv[1]

with open(sys.argv[2],"r") as f:
    para= json.load(f)

# this would be "best" for train_test runs that record best results
# or it would be "test_results" for separate test runs from validation run best checkpoints
secondary = sys.argv[3]

if len(sys.argv) > 4:
    basename2 = sys.argv[4]
else:
    basename2 = None
N_avg = 5
s= np.product(para["split"])
res_names = [ "id", "total number epochs", "final train correct", "final train loss", "final valid correct", "final valid loss", "epoch of best valid", "train correct at best valid", "train loss at best valid", "best valid correct", "valid loss at best valid", "time per epoch" ]
opt_col = 9 # that is best validation performance
the_dt= [ 1, 2, 5, 10, 20 ]

results= load_train_test(basename,s,N_avg,secondary)
if basename2 is not None:
    results2= load_train_test(basename2,s,N_avg,secondary)

cond = ["no delay, homo", "no delay, hetero", "delay, homo", "delay, hetero" ]

for dt in range(3):
    fig, ax = plt.subplots(1,4,sharey=True,figsize=((8,2.5)))
    # for delay line or not
    xmn = 1.0
    xmx = 0.0
    for i in range(2):
        # for hidden neuron type 
        for j in range(2):
            x= np.zeros((4,5))
            y= np.zeros((4,5))
            # blend no/yes
            for b in range(2):
                # shift no/yes
                for sh in range(2):
                    for sz in range(5):
                        id = dt*320+sz*64+i*32+j*16+b*8+sh*4
                        x[b*2+sh,sz]= np.mean(results[9,id:id+4])
                        y[b*2+sh,sz]= np.std(results[9,id:id+4])
            for k in range(4):
                print(x[k,:])
                #plt.plot(x[k,:])
                ax[i*2+j].errorbar(np.arange(5),x[k,:],y[k,:])
            xmn= min(np.amin(x.flatten()-y.flatten()),xmn)
            xmx= max(np.amax(x.flatten()+y.flatten()),xmx)
            ax[i*2+j].set_xticks([ 0, 1, 2, 3, 4 ])
            ax[i*2+j].set_xticklabels([ 64, 128, 256, 512, 1024 ])
            ax[i*2+j].set_yticks(np.arange(0,1.0,0.02), minor=True)
            ax[i*2+j].set_yticks(np.arange(0.5,1.01,0.1))
            ax[i*2+j].tick_params(which='minor', length=0)
            ax[i*2+j].grid(which='minor',alpha=0.3)
            ax[i*2+j].grid(which='major',alpha=1,axis="y")
            ax[i*2+j].spines['top'].set_visible(False)
            ax[i*2+j].spines['right'].set_visible(False)
            ax[i*2+j].set_title(cond[i*2+j])
    ax[0].set_ylim([xmn*0.95, xmx*1.05])
    ax[0].set_xlabel(" ")
    ax[0].set_ylabel("test accuracy")

    plt.tight_layout()
    plt.savefig(f"{basename}_augment_hidN_dt{the_dt[dt]}.pdf")

    # investigate whether heterogeneous neurons help
    b = 1  # take blend on
    sh = 1 # take shift on
    x = []
    y = []
    # any delay off/on
    for i in range(2):
        # any N_hid
        for sz in range(5):
            # two points for hetero or not
            for j in range(2):
                id= dt*320+sz*64+i*32+j*16+b*8+sh*4
                x.append(np.mean(results[9,id:id+4]))
                y.append(np.std(results[9,id:id+4]))
    x= np.array(x).reshape((-1,2))
    y= np.array(y).reshape((-1,2))
    if basename2 is not None:
        x2 = []
        y2 = []
        # any delay off/on
        for i in range(2):
            # any N_hid
            for sz in range(5):
                # use homo & compare no tau learn, and tau learning
                j = 0
                id= dt*320+sz*64+i*32+j*16+b*8+sh*4
                x2.append(np.mean(results[9,id:id+4]))
                x2.append(np.mean(results2[9,id:id+4]))
                y2.append(np.std(results[9,id:id+4]))
                y2.append(np.std(results2[9,id:id+4]))
        x2= np.array(x2).reshape((-1,2))
        y2= np.array(y2).reshape((-1,2))
        x3 = []
        y3 = []
        # any delay off/on
        for i in range(2):
            # any N_hid
            for sz in range(5):
                # use hetero & compare no tau learn, and tau learning
                j = 1
                id= dt*320+sz*64+i*32+j*16+b*8+sh*4
                x3.append(np.mean(results[9,id:id+4]))
                x3.append(np.mean(results2[9,id:id+4]))
                y3.append(np.std(results[9,id:id+4]))
                y3.append(np.std(results2[9,id:id+4]))
        x3= np.array(x3).reshape((-1,2))
        y3= np.array(y3).reshape((-1,2))
    plt.figure(figsize=(6,4))
    idx= np.argsort(x[:,0])
    for i in range(x.shape[0]):
        plt.errorbar(np.arange(2)+i*0.03,x[idx[i],:],y[idx[i],:],lw=1)
    if basename2 is not None:
        idx= np.argsort(x2[:,0])
        for i in range(x2.shape[0]):
            plt.errorbar(np.arange(2)+2+i*0.03,x2[idx[i],:],y2[idx[i],:],lw=1)
        idx= np.argsort(x3[:,0])
        for i in range(x3.shape[0]):
            plt.errorbar(np.arange(2)+4+i*0.03,x3[idx[i],:],y3[idx[i],:],lw=1)
    ax= plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    xmx = ax.get_ylim()[1]
    #ax.text(0.0,xmx,"homo \n no learn",fontsize=12)
    #ax.text(1.0,xmx,"hetero \n no learn",fontsize=12)
    #if basename2 is not None:
    #    ax.text(2.0,xmx,"homo \n no learn",fontsize=12)
    #    ax.text(3.0,xmx,"homo \n learn",fontsize=12)
    #ax.set_xlim([ -0.3, 1.3 ])
    ax.set_ylabel("test accuracy")
    plt.tight_layout()
    plt.savefig(f"{basename}_homo_hetero_dt{the_dt[dt]}.pdf")

# Let's now have a look at using large timesteps
j = 1 # consider hetero now
x = np.zeros((5,5))
y = np.zeros((5,5))
for b in range(2):
    for sh in range(2):
        # any delay off/on
        for i in range(2):
            # any N_hid
            for sz in range(5):
                for dt in range(5):
                    id= dt*5*64+sz*64+i*32+j*16+b*8+sh*4
                    x[dt,sz] = np.mean(results[9,id:id+4])
                    y[dt,sz] = np.std(results[9,id:id+4])
            plt.figure(figsize=(2.9,2.3))
            vmn = np.min(x.flatten())*0.95
            vmx = np.max(x.flatten())*1.05
            plt.imshow(x,vmin=vmn,vmax=vmx,cmap="hot")
            ax= plt.gca()
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
            #ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_yticklabels([ 1,2,5,10,20])
            ax.set_xticks(range(5))
            ax.set_xticklabels([64,128,256,512,1024],rotation=60)
            ax.set_xlabel(" ")
            ax.set_ylabel(" ")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{basename}_dt_nhid_delay{i}_shift{sh}_blend{b}.pdf")

# Let's now have a look at using large timesteps wrt runtimes
j = 1 # consider hetero now
x = np.zeros((5,5))
y = np.zeros((5,5))
for b in range(2):
    for sh in range(2):
        # any delay off/on
        for i in range(2):
            # any N_hid
            for sz in range(5):
                for dt in range(5):
                    id= dt*5*64+sz*64+i*32+j*16+b*8+sh*4
                    x[dt,sz] = np.mean(results[11,id:id+4])
                    y[dt,sz] = np.std(results[11,id:id+4])
            plt.figure(figsize=(2.9,2.3))
            vmn = np.min(x.flatten())*0.95
            vmx = np.max(x.flatten())*1.05
            plt.imshow(x,vmin=vmn,vmax=vmx,cmap="hot")
            ax= plt.gca()
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
            #ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_yticklabels([ 1,2,5,10,20])
            ax.set_xticks(range(5))
            ax.set_xticklabels([ 64,128,256,512,1024],rotation=60)
            ax.set_xlabel(" ")
            ax.set_ylabel(" ")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{basename}_dt_nhid_runtime_delay{i}_shift{sh}_blend{b}.pdf")

plt.show()
