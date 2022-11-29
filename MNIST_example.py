import numpy as np
import matplotlib.pyplot as plt
import mnist

p= {
    "TRIAL_MS": 20.0,
    "DT_MS": 1.0
    }

def spike_time_from_gray(t):
    return (255.0-t)/255.0*(p["TRIAL_MS"]-4*p["DT_MS"])+2*p["DT_MS"]   # make sure spikes are two timesteps within the presentation window

X = mnist.train_images()
Y = mnist.train_labels()

x=X[0,:]

ax= []
fig=plt.figure(figsize= (2,8))
ax.append(plt.subplot2grid(shape=(4,1), loc= (0,0), rowspan= 1))
ax.append(plt.subplot2grid(shape=(4,1), loc= (1,0), rowspan= 3))

ax[0].imshow(256-x,cmap='gray')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel("28")
ax[0].set_ylabel("28")
ix= x > 1
x= x[ix]
x= spike_time_from_gray(x)
ix= ix.flatten()
tid= np.arange(len(ix))[ix]

ax[1].scatter(x, tid, marker='|', s= 4, color='k')
ax[1].set_xlabel("t (ms)")
ax[1].set_ylabel("neuron id")
ax[1].set_ylim([0, 784])
plt.tight_layout()

plt.savefig("MNIST_example.png", dpi=300)
plt.show()
