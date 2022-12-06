import numpy as np
import matplotlib.pyplot as plt

g= np.load("test43_w_hidden_output_last.npy")

plt.figure()
plt.hist(g)

print(g.shape)

#plt.figure()
#plt.hist(g[::32])
#plt.figure()
#plt.hist(g[1::32])
#plt.figure()
#plt.hist(g[19::32])
mn= [ np.mean(g[i::32]) for i in range(20) ]
print(mn)
plt.figure()
plt.barh(range(len(g[0::32])),g[0::32])

g2= np.load("test43_w_input_hidden_last.npy")

g2=g2.reshape((700,1))

hig= np.sum(g2,axis=0)

plt.figure()
plt.imshow(g2)

plt.figure()
plt.barh(np.arange(len(hig)),hig)

plt.show()
