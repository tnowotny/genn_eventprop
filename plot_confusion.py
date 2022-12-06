import numpy as np
import matplotlib.pyplot as plt

import sys

name= sys.argv[1]
epoch= int(sys.argv[2])
for ph in ["train","eval"]:
    data= np.load(name+"_confusion"+ph+".npy")
    plt.figure()
    plt.title(ph)
    plt.imshow(data[epoch])

plt.show()
