import numpy as np
import matplotlib.pyplot as plt
import sys

id= np.load(sys.argv[1]+"_"+sys.argv[2]+"_spike_ID.npy")
t= np.load(sys.argv[1]+"_"+sys.argv[2]+"_spike_t.npy")
lbl= np.load(sys.argv[1]+"_spk_lbl.npy")
print(lbl)
print(t) 
"""
dt= t[1:]-t[:-1]
cut= np.where(dt < 0)
cut= cut[0]
cut= np.hstack([ [0], cut, [len(dt)]])
cut_t= [ t[cut[i]:cut[i+1]] for i in range(len(cut)-1)]
cut_id= [ id[cut[i]:cut[i+1]] for i in range(len(cut)-1)]

print(cut)
if len(cut) > 1:
    fig, ax= plt.subplots(len(cut_t), 1)
    for lax, lt, lid in zip(ax,cut_t,cut_id):
        lax.scatter(lt, lid, s= 0.1)
else:
"""
plt.figure()
plt.scatter(t,id,s=0.2)
        
plt.show()
