import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from load_specific_scans import load_SSC_test_19
from load_traintest_SHD_final import load_traintest_SHD_final
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

"""
This script was used for generating figures 3 and 4 of the publication. Final assembly was in powerpoint
"""

if len(sys.argv) != 1:
    print("Usage: python plot_SHD_paper_augment_hidN_2.py")
    exit(1)

#which = "SHD"
which = "SSC"
N_avg = 5
res_names = [ "id", "total number epochs", "final train correct", "final train loss", "final valid correct", "final valid loss", "epoch of best valid", "train correct at best valid", "train loss at best valid", "best valid correct", "valid loss at best valid", "time per epoch" ]
opt_col = 9 # that is best validation performance

if which == "SHD":
    results= load_traintest_SHD_final(N_avg)
if which == "SSC":
    results= load_SSC_test_19(N_avg)
    
dt= [ 1, 2 ]
num_hidden = [ 64, 128, 256, 512, 1024 ]
cond = ["homo,const", "homo,learn", "hetero,const", "hetero,learn" ]

# different dt
for i in range(2):
    act = []
    fig, ax = plt.subplots(1,4,sharey=True,figsize=(9,3.2))
    xmn = 1.0
    xmx = 0.0
    # hiddenneuron type
    for n in range(2):
        # train tau
        for o in range(2):
            x= np.zeros((8,5))
            y= np.zeros((8,5))
            # for delay line or not
            for k in range(2):
                # shift
                for l in range(2):
                    # blend
                    for m in range(2):
                        # for different hidden sizes
                        for j in range(5):
                            id = ((((((i*5+j)*2+k)*2+l)*2+m)*2+n)*2+o)*8
                            x[(k*2+l)*2+m,j] = np.mean(results[9,id:id+8])
                            y[(k*2+l)*2+m,j] = np.std(results[9,id:id+8])
            for curve in range(8):
                print(x[curve,:])
                act.append(ax[n*2+o].errorbar(np.arange(5),x[curve,:],y[curve,:]))
            xmn= min(np.amin(x.flatten()-y.flatten()),xmn)
            xmx= max(np.amax(x.flatten()+y.flatten()),xmx)
            ax[n*2+o].set_xticks([ 0, 1, 2, 3, 4 ])
            ax[n*2+o].set_xticklabels(num_hidden)
            ax[n*2+o].set_yticks(np.arange(0,1.0,0.02), minor=True)
            ax[n*2+o].set_yticks(np.arange(0.5,1.01,0.1))
            ax[n*2+o].tick_params(which='minor', length=0)
            ax[n*2+o].grid(which='minor',alpha=0.3)
            ax[n*2+o].grid(which='major',alpha=1,axis="y")
            ax[n*2+o].spines['top'].set_visible(False)
            ax[n*2+o].spines['right'].set_visible(False)
            ax[n*2+o].set_title(cond[n*2+o])
    ax[0].set_ylim([xmn*0.95, xmx*1.05])
    ax[0].set_xlabel(" ")
    ax[0].set_ylabel("test accuracy")
    fig.legend(act[:8],["plain", "blend", "shift", "shift+blend", "delay", "delay+blend", "delay+shift", "delay+shift+blend"],loc="lower center", ncol=4)

    fig.tight_layout(pad=0.1, rect=[0.0, 0.2, 1.0, 1.0])
    if which == "SHD":
        namefrac= "scan_JUWELS_48"
    if which == "SSC":
        namefrac= "scan_SSC_JUWELS_19"
    plt.savefig(f"{namefrac}_augment_hidN_dt{dt[i]}.pdf")


# Let's now have a look at runtimes and parameter counts

fig, ax = plt.subplots(1,2,figsize=((9,4.5)))
symbs = ['^', 'o', 'v', 's','<','p','>','*',"1","2","3","4","+","x","P","X"]
sz = np.asarray([ 10, 10, 10, 10, 10, 10, 10, 12, 20, 20, 20, 20, 16, 12, 12, 12 ])*2
# different dt
for i in range(2):
    act = []
    xmn = 1.0
    xmx = 0.0
    # hiddenneuron type
    x = np.zeros((32,5))
    y = np.zeros((32,5))
    # for delay line or not
    for k in range(2):
        # shift
        for l in range(2):
            # blend
            for m in range(2):
                # heterogeneous
                for n in range(2):
                    # train tau
                    for o in range(2):
                        # for different hidden sizes
                        for j in range(5):
                            id = ((((((i*5+j)*2+k)*2+l)*2+m)*2+n)*2+o)*8
                            x[(((k*2+l)*2+m)*2+n)*2+o,j] = np.mean(results[9,id:id+8])
                            y[(((k*2+l)*2+m)*2+n)*2+o,j] = np.mean(results[11,id:id+8])

    for j in range(5):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    idx = ((k*2+l)*2+m)*4
                    print(x[idx:idx+4,j])
                    ax[0].scatter(x[idx:idx+4,j],y[idx:idx+4,j]*1000,marker=symbs[i*8+idx//4],color=f"C{j}",s=sz[i*8+idx//4])
#ax[0].set_xlim([0,1])
ax[0].set_ylim([0,13])
ax[0].set_ylabel("wall clock time / sample (ms)")
ax[0].set_xlabel("test accuracy")
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

npmax = 0
for i in range(2):
    xmn = 1.0
    xmx = 0.0
    # hiddenneuron type
    x = np.zeros((32,5))
    y = np.zeros((32,5))
    # for delay line or not
    for k in range(2):
        # shift
        for l in range(2):
            # blend
            for m in range(2):
                # heterogeneous
                for n in range(2):
                    # train tau
                    for o in range(2):
                        # for different hidden sizes
                        for j in range(5):
                            id = ((((((i*5+j)*2+k)*2+l)*2+m)*2+n)*2+o)*8
                            x[(((k*2+l)*2+m)*2+n)*2+o,j] = np.mean(results[9,id:id+8])
                            nin = 700 if k == 0 else 7000
                            nhid = num_hidden[j]
                            nout = 20
                            npara = nin*nhid+nhid*nhid+nhid*nout
                            if o == 1:
                                npara += nhid*2
                            if npara > npmax:
                                npmax = npara
                            y[(((k*2+l)*2+m)*2+n)*2+o,j] = npara
                            
    print(f"max para: {npmax}")
    for j in range(5):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    idx = ((k*2+l)*2+m)*4
                    print(x[idx:idx+4,j])
                    ax[1].scatter(x[idx:idx+4,j],y[idx:idx+4,j]/1e6,marker=symbs[i*8+idx//4],color=f"C{j}",s=sz[i*8+idx//4])
#ax[1].set_xlim([0,1])
#ax[1].set_ylim([0,15])
ax[1].set_ylabel("number of parameters (million)")
ax[1].set_xlabel("test accuracy")
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
hdls = []
#labels = ["plain","blend","shift","shift+blend", "delay", "delay+blend", "delay+shift", "delay+shift+blend" ]
labels = ["    ", "    ", "    ", "        ", "    ", "      ", "       ", "      "]
labels.extend(labels)
for i,smb in enumerate(symbs):
    hdls.append(Line2D([0], [0], label=labels[i], marker=smb, markersize=sz[i]/5, color='k',linestyle=""))
for i in range(5):
    hdls.append(mpatches.Patch(color=f"C{i}", label=str(num_hidden[i])))
for i in range(3):
    hdls.append(mpatches.Patch(color="w", label=""))
    
hdls2= hdls.copy()
for j in range(3):
    hdls2[j::3]=hdls[j*8:(j+1)*8]
        
fig.legend(handles=hdls2,loc="lower center", ncol=8,fontsize=8.5)
fig.tight_layout(pad=0, rect=[0.0, 0.25, 1.0, 1.0])
plt.savefig(f"{namefrac}_wallclock_npara.pdf")


plt.show()
