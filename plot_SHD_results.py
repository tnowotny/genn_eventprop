import numpy as np
import matplotlib.pyplot as plt
import sys

res= np.loadtxt(sys.argv[1]+"_results.txt", dtype= np.float32)
plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
plt.plot(1-res[:,1])
plt.plot(1-res[:,3])
#plt.yscale("log")
#plt.xscale("log")
#plt.ylim(0.0001,1)
plt.title("% Error")
plt.subplot(2,2,3)
plt.plot(res[:,2])
plt.plot(res[:,4])
#plt.legend(["train 0.95/0.9995", "eval 0.95/0.9995", "train 0.9/0.999", "eval 0.9/0.999", "train 0.99/0.9999", "eval 0.99/0.9999", "train 0.995/0.99995", "eval 0.995/0.99995", "train DT=1ms, 0.99/0.9999,pDrop= 0.1", "eval DT=1ms, 0.99/0.9999,pDrop= 0.1","train n_hid= 1000","eval n_hid= 1000"])
plt.yscale("log")
#plt.xscale("log")
#plt.ylim(0.001,20)
plt.title("Loss")
plt.subplot(2,2,2)
plt.plot(res[:,9])
plt.errorbar(np.arange(res.shape[0]),res[:,9],res[:,10])
plt.plot(res[:,11])
plt.plot(res[:,12])
plt.title("sNSum")
plt.subplot(2,2,4)
plt.plot(res[:,13])
plt.title("n_rewire")

plt.show()
plt.savefig("test2.png", dpi= 300)

