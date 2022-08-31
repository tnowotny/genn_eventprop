import numpy as np
import matplotlib.pyplot as plt
import sys

Y= [ 1, 6, 0, 14 ]
fig, ax= plt.subplots(3,1)
V= np.load("test43_Voutput.npy")
#V2= np.load("test43_Ioutput.npy")

print(V.shape)
ax[0].plot(V[:,0,:])
#ax[0].plot(V[:,0,Y[:-1]],'.')
#ax[1].plot(V2[:,0,:])
#ax[1].plot(V[:,1,0:2])
#ax[2].plot(V[:,0,:]-V2[:,0,:])
#ax[1].set_yscale('log')
#ax[1].plot(V2[:,0,Y[:-1]],'.')
plt.show()
