import numpy as np
import matplotlib.pyplot as plt
import sys

basename= sys.argv[1]

id= np.load("test_"+basename+"_hidden_spike_ID.npy")
t= np.load("test_"+basename+"_hidden_spike_t.npy")
trgt= 0
dur= 1400
total_sN= []

plt.figure()
plt.scatter(t,id)

for ep in range(2):
    for tr in range(8):
        tme= ((ep*8)+tr)*32*1400
        sidx= np.where(np.logical_and(t > tme,t < tme+32*1400))
        lid= id[sidx]
        lt= t[sidx]
        plt.figure()
        plt.scatter(lt,lid,s=0.1)

plt.show()
