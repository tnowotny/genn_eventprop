import numpy as np

x= np.loadtxt("test_axe3_allresult.txt")
mn= np.mean(x[:,0])
st= np.std(x[:,0])
print(f"Mean {mn}, std {st}")
