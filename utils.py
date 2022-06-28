import numpy as np

"""
this is an augmentation for SHD spoken digits
spiking data is shifted by a random number of neurons along the neuron ID axis
"""
def random_shift(X,rng,max_shift):
    for x in X:
        shift= rng.uniform(-max_shift,max_shift)
        x["x"]= np.maximum(np.minimum(x["x"]+shift,699),0)

"""
dilate or compress time
"""
def random_dilate(X,rng,min_factor,max_factor):
    mn= np.log(min_factor)
    mx= np.log(max_factor)
    for x in X:
        fac= np.exp(rng.uniform(mn,mx))
        x["t"]= np.minimum(x["t"]*fac,1400)
