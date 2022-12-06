import numpy as np
import matplotlib.pyplot as plt

st= np.load("test105_hidden_spike_t.npy")
ID= np.load("test105_hidden_spike_ID.npy")
lb= np.load("test105_spk_lbl.npy")
print(lb)
print(st)
plt.figure()
plt.plot(st)
plt.figure()
plt.scatter(st,ID)
plt.show()
