import numpy as np
import matplotlib.pyplot as plt
import sys

name= sys.argv[1]
name2= sys.argv[2]

Y= [ 1, 6, 0, 14 ]
fig, ax= plt.subplots(2,1)
V= np.load(name)
V2= np.load(name2)

print(V.shape)
ax[0].plot(V[:,0,:])
ax[0].plot(V[:,0,Y[:-1]],'.')
ax[1].plot(V2[:,0,:])
ax[1].plot(V2[:,0,Y[:-1]],'.')
plt.show()
