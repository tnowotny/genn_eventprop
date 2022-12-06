import tables
import numpy as np
import matplotlib.pyplot as plt

ex= 1
hdf5_file_path= "/its/home/tn41/data/hdspikes/shd_train.h5"
 
fileh = tables.open_file(hdf5_file_path, mode='r')
units = fileh.root.spikes.units
times = fileh.root.spikes.times
labels = fileh.root.labels
speaker= fileh.root.extra.speaker

speaker= np.array(speaker)
labels= np.array(labels)
spks= np.unique(speaker)
idx= []
plots= []
cnt= np.arange(len(speaker))
for s in spks:
    tidx= cnt[np.logical_and(speaker == s, labels == ex)]
    idx.append(tidx)
    print(tidx.shape)
    plt.figure()
    for i in range(16):
        ax = plt.subplot(4,4,i+1)
        ax.scatter(times[tidx[i]],700-units[tidx[i]], color="k", alpha=0.33, s=2)
        ax.set_title("Speaker {}, Label {}".format(s, labels[tidx[i]]))
        ax.axis("off")

plt.show()
