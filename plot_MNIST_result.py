import numpy as np
import matplotlib.pyplot as plt

res= np.loadtxt("test1_results.txt", dtype= np.float32)

plt.figure
plt.subplot(2,1,1)
plt.plot(res[:,1])
plt.plot(res[:,3])
plt.title("% Correct")
plt.subplot(2,1,2)
plt.plot(res[:,2])
plt.plot(res[:,4])
plt.title("Loss")
plt.savefig("test1.png", dpi= 300)
plt.show()
