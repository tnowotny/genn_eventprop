import numpy as np
import matplotlib.pyplot as plt
import sys

basename= sys.argv[1]

id= np.load("test_"+basename+"_hidden_spike_ID.npy")
t= np.load("test_"+basename+"_hidden_spike_t.npy")
g= np.load("test_"+basename+"_w_hidden_output_last.npy")
trgt= 0
dur= 1400

fig, ax= plt.subplots(1,2,sharey=True)
ax[0].scatter(t,id,s=0.1)
left= int(t[-1]-dur)//dur*dur
ax[0].set_xlim(left,left+dur)
ax[0].set_ylim(0,25)
ax[0].set_xlabel("time (ms)")
ax[0].set_ylabel("hidden neuron")

# plot the synaptic weights into the last trial's correct output neuron
ax[1].barh(range(len(g[trgt::32])),g[trgt::32])
ax[1].set_xlabel("w to output 0 (nS)")
mn= [ np.mean(g[i::32]) for i in range(20) ]
print(mn)

# consider the last batch
ID_last= id[t >= left]
print(ID_last.shape)
# number of spikes in each hidden neuron
sNlast= [ len(np.where(ID_last == i)[0]) for i in range(256) ]
print(sNlast)
plt.savefig("../theory/test_"+basename+"_spikes_weights.png",dpi=300)

fig, ax= plt.subplots(1,2,sharey=True)
ax[0].barh(range(len(sNlast)),sNlast)
ax[0].set_ylim(0,25)
ax[0].set_xlabel("number of spikes last trial (unitless)")
ax[0].set_ylabel("hidden neuron")

ax[1].barh(range(len(g[trgt::32])),g[trgt::32])
ax[1].set_xlabel("w to output 0 (nS)")

allg= [ g[i::32] for i in range(20)]
x= np.vstack([allg,sNlast])
#print(x.T)
print(np.corrcoef(x))
plt.savefig("../theory/test_"+basename+"_rates_weights.png",dpi=300)

plt.figure()
plt.set_cmap("jet")
plt.imshow(np.corrcoef(x),vmin=-1,vmax=1)
plt.colorbar()
plt.xlabel("output neuron ID (0-19) / hidden spike rate (20)")
plt.ylabel("output neuron ID (0-19) / hidden spike rate (20)")
plt.title("correlations between output neuron incoming weight vector \n and spike rate of hidden neurons")
plt.savefig("../theory/test_"+basename+"_correlations.png",dpi=300)
plt.show()
