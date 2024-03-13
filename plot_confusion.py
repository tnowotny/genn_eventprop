import numpy as np
import matplotlib.pyplot as plt

import sys

name= sys.argv[1]
epoch= int(sys.argv[2])
for ph in ["train","eval"]:
    data= np.load(name+"_confusion_"+ph+".npy")
    plt.figure()
    plt.title(ph)
    plt.imshow(data[epoch])
    print(data[epoch])
    x= data[epoch]
    for i in range(x.shape[0]):
        x[i,i] = 0
    plt.imshow(x)
    print(np.sum(x.flatten()))
    print(np.amax(x.flatten()))
plt.show()
