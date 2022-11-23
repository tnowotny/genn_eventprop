import numpy as np
import matplotlib.pyplot as plt
import os
import tables
import random

cache_dir=os.path.expanduser("~/data")
cache_subdir="SHD"
print("Using cache dir: %s"%cache_dir)
num_input= 700
hdf5_file_path= 'data/SHD/shd_train.h5'
fileh= tables.open_file(hdf5_file_path, mode='r')
units= fileh.root.spikes.units
times= fileh.root.spikes.times
label= fileh.root.labels
fig,ax= plt.subplots(2,2,sharex= True, sharey= True, figsize= (8,8))

lbl= ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

random.seed(1)

for i in range(4):
    tid= random.randint(0,999)
    x= units[tid]
    t= times[tid]
    tax= ax[i//2, i%2]
    
    tax.scatter(t*1000, 700-x, marker='|', s= 4, color='k')
    tax.set_title(lbl[label[tid]])
    if i//2 == 1:
        tax.set_xlabel("t (ms)")
    if i%2 == 0:
        tax.set_ylabel("neuron id")
plt.tight_layout()
plt.savefig("SHD_example.png", dpi= 300)
plt.show()
