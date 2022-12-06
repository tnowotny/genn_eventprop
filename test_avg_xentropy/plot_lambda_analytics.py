import numpy as np
import matplotlib.pyplot as plt
import sys

"""
Plot analysis of the gradient flow from the correct output neuron in the backward phase to the 
hidden neurons. Look especially at the most active hidden neurons.
Uses test_axe1 output.
"""

verbose= False

def mprint(x):
    if verbose:
        print(x)

id= np.load("test_axe1_hidden_spike_ID.npy")
t= np.load("test_axe1_hidden_spike_t.npy")
labels= np.load("test_axe1_labels_train.npy")
pred= np.load("test_axe1_predictions_train.npy")
lbdVout= np.load("test_axe1_lambda_Voutput.npy")
lbdIout= np.load("test_axe1_lambda_Ioutput.npy")
lbdVhid= np.load("test_axe1_lambda_Vhidden.npy")

trgt= 0
dur= 1400
n_batch= 32
xlm= [0, 800]
xtk= np.array([ 0, 200,400, 600, 800 ])
lnex= xlm
lney= [ 0, 0]
total_sN= []


# find out which input pattern occurs frequently in trials 0 and 1:

ep= 0
shown= labels[0,:2*n_batch]
mprint(shown)
shist= np.histogram(shown, bins=20, range= (0,19))
mprint(shist[0])
lbl= np.argmax(shist[0])  # the label that was shown the most often
print(f"label is {lbl}")
tr= np.where(shown == lbl)[0]  # for whatever reason np.where returns a tuple with one array entry
mprint(tr)
aNs= []
for tr_b in tr:   # got through the trials with the chosen label
    mprint(tr_b)
    tme= (ep*2*n_batch+tr_b)*dur # start time of each input
    mprint(tme)
    lid= id[np.logical_and(t >= tme,t < tme+dur)]
    lt= t[np.logical_and(t >= tme,t < tme+dur)]
    Ns= np.histogram(lid, bins= 256, range= (0,255))
    aNs.append(Ns[0])
aNs= np.vstack(aNs)
mprint(aNs)
avgNs= np.mean(aNs, axis= 0)
mprint(avgNs)
top10= np.argsort(avgNs)[:10]
mprint(top10)
mprint(avgNs[top10])

for tr_b in tr:
    mprint(tr_b)
    tme= (ep*2*n_batch+tr_b)*dur # start time of each input
    mprint(tme)
    plt.figure(figsize= (5.5,5))
    lid= id[np.logical_and(t >= tme,t < tme+dur)]
    lt= t[np.logical_and(t >= tme,t < tme+dur)]
    ax0= plt.subplot2grid(shape=(5,1), loc= (0,0), rowspan= 4)
    ax1= plt.subplot2grid(shape=(5,1), loc= (4,0), rowspan= 1)
    strt= (ep*2*n_batch+tr_b)//n_batch*dur+dur  # show lambdas from the next trial
    ibatch= tr_b % n_batch
    mprint(ibatch)
    mprint(strt)
    mprint(lbdVout.shape)
    lV= lbdVout[strt+dur:strt:-1,ibatch,lbl]
    ax1.plot(lV*1e5, lw=0.5)
    lI= lbdIout[strt+dur:strt:-1,ibatch,lbl]
    ax1.plot(lI*1e5, lw=0.5)
    #ax1.plot((lV-lI)*1e5, lw=0.5)
    for i in range(256):
        tspk= lt[lid == i]
        tspk= tspk-tme
        if i in top10:
            clr= 'r'
        else:
            clr= 'k'
        ax0.scatter(tspk,i*np.ones(len(tspk)),marker='|',s=1,linewidths=0.4,color=clr)
    ax0.set_xlim([0, 1400])
    ax0.set_xticklabels([])
    ylow= np.min(lI[1:dur-100]*1e5)
    yhigh= np.max(lI*1e5)
    yhigh= yhigh+0.1*(yhigh-ylow)
    ax1.set_ylim([ylow, yhigh])
    ax1.set_xlim([0, dur])
    ax1.set_xlabel("time (ms)")
    ax0.set_ylabel("neuron ID (unitless)")
    ax1.legend(["lambda_V","lambda_I"],fontsize=8)
    plt.savefig("test_axe1_lbdfig_trb_"+str(tr_b)+".png", dpi=300)

    fig, ax= plt.subplots(11,1,sharex=True,figsize= (2.5,5))
    dVI= (lV-lI)*1e8
    ax[0].plot(dVI, lw= 0.5)
    ax[0].set_xlim(xlm)
    ax[0].set_xticks(xtk, xtk)
    ylow= np.min(dVI[int(xlm[0]):int(xlm[1])])
    yhigh= np.max(dVI[int(xlm[0]):int(xlm[1])])
    mprint(ylow)
    mprint(yhigh)
    ax[0].set_ylim([ ylow*1.2, yhigh*1.2] )
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    yl= []
    yh= []
    top10= np.sort(top10)
    incsum= []
    for i in range(10):
        tspk= lt[lid == top10[i]]
        tspk= np.array(tspk-tme, dtype= int)
        mprint(i)
        mprint(lV)
        inc= np.array(lV[tspk]-lI[tspk])*1e8
        ax[i+1].bar(tspk, inc, width=4)
        #print(dir(ax[0]))
        ax[i+1].spines['top'].set_visible(False)
        ax[i+1].spines['right'].set_visible(False)
        ax[i+1].set_xlim(xlm)
        ax[i+1].plot(lnex,lney,'k',lw= 1)
        if len(inc) > 0:
            yl.append(np.min(inc))
            yh.append(np.max(inc))
        incsum.append(np.sum(inc))
    if len(yl) > 0:
        ylow= np.min(yl)*1.2
    else:
        ylow= 0
    if len(yh) > 0:
        yhigh= np.max(yh)*1.2
    else:
        yhigh= dur
    for i in range(10):
        ax[i+1].text(lnex[1]-200, 0.6*ylow, "%.2g" % incsum[i])
        ax[i+1].set_ylim([ylow, yhigh])
    plt.savefig("test_axe1_lV-lI_fig_trb_"+str(tr_b)+".png", dpi=300)

