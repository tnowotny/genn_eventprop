import numpy as np
import matplotlib.pyplot as plt
import sys

"""
Analyse the failing learning dynamics with avg_xentropy:
compare how hidden representations change in test_axe2
"""

verbose= False

def mprint(x):
    if verbose:
        print(x)

id= np.load("test_axe2_hidden_spike_ID.npy")
t= np.load("test_axe2_hidden_spike_t.npy")
labels= np.load("test_axe2_labels_train.npy")
pred= np.load("test_axe2_predictions_train.npy")
lbdVout= np.load("test_axe2_lambda_Voutput.npy")
lbdIout= np.load("test_axe2_lambda_Ioutput.npy")
lbdVhid= np.load("test_axe2_lambda_Vhidden.npy")

mprint(labels.shape)
mprint(pred.shape)

trgt= 0
dur= 1400
n_batch= 32
n_trial= 8
xlm= [0, 800]
xtk= np.array([ 0, 200,400, 600, 800 ])
lnex= xlm
lney= [ 0, 0]
total_sN= []

fig, ax= plt.subplots(1,3,sharey=True,figsize=(8,4))
g= np.load("test_axe2_w_hidden_output_last.npy")

# plot the synaptic weights into the last trial's correct output neuron
cg= g[trgt::20]
idx= np.argsort(cg)
idx2= np.zeros(len(idx))
idx2[idx]= np.arange(len(idx))
mprint(idx2)
ax[0].barh(range(len(cg)),cg[idx])
ax[0].set_xlabel("w to output 0 (uS)")
ax[0].set_ylabel("hidden neuron")
mn= [ np.mean(g[i::20]) for i in range(20) ]

# plot and example spike raster
ep= 29
tr= 7
ib= 0
tme= ((ep*n_trial+tr)*n_batch+ib)*dur
lid= id[np.logical_and(t >= tme,t < tme+dur)]
lt= t[np.logical_and(t >= tme,t < tme+dur)]
ax[1].scatter(lt-tme,idx2[lid],marker='|',s=1,linewidths=0.4,)
#left= int(t[-1]-dur)//dur*dur
#ax[0].set_xlim(left,left+dur)
#ax[0].set_ylim(0,25)
ax[1].set_xlabel("time (ms)")

# build up spiking statistics across the last batch
b_dur= dur*n_batch
lid= id[np.logical_and(t >= tme,t < tme+b_dur)]
lt= t[np.logical_and(t >= tme,t < tme+b_dur)]
Ns= np.histogram(lid, bins= 256, range= (0,255))
Ns= Ns[0]/b_dur*1000
ax[2].barh(range(len(Ns)),Ns[idx])
ax[2].set_xlabel("avg rate in last batch (Hz)")
plt.savefig("push_pull_illustration.png", dpi=300)

cc= np.corrcoef(np.vstack([cg,Ns]))
print(cc)
plt.show()
