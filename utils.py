import numpy as np

"""
this is an augmentation for SHD spoken digits
spiking data is shifted by a random number of neurons along the neuron ID axis
"""
def random_shift(X,rng,max_shift):
    for i in range(len(X)):
        shift= int(rng.uniform(-max_shift,max_shift))
        X[i]["x"]= X[i]["x"]+shift
        if shift > 0:
            idx= X[i]["x"] < 700
            X[i]["x"]= X[i]["x"][idx]
            X[i]["t"]= X[i]["t"][idx]
        else:
            idx= X[i]["x"] >= 0
            X[i]["x"]= X[i]["x"][idx]
            X[i]["t"]= X[i]["t"][idx]
            
    return X


"""
jitter the ID of neurons that spiked as an augmentation (see SHD paper)
"""
def ID_jitter(X,rng,sigma):
    for i in range(len(X)):
        shift= rng.standard_normal(len(X[i]["x"]))*sigma
        X[i]["x"]= X[i]["x"]+shift
        idx= np.logical_and(X[i]["x"] < 700, X[i]["x"] >= 0)
        X[i]["x"]= X[i]["x"][idx]
        X[i]["t"]= X[i]["t"][idx]
    return X
    

"""
dilate or compress time
"""
def random_dilate(X,rng,min_factor,max_factor):
    mn= np.log(min_factor)
    mx= np.log(max_factor)
    for i in range(len(X)):
        fac= np.exp(rng.uniform(mn,mx))
        X[i]["t"]= X[i]["t"]*fac
        idx= X[i]["t"] < 1.4
        X[i]["t"]= X[i]["t"][idx]
        X[i]["x"]= X[i]["x"][idx]
    return X
