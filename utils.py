import numpy as np

"""
this is an augmentation for SHD spoken digits
spiking data is shifted by a random number of neurons along the neuron ID axis
"""
def random_shift(X,rng,max_shift,p):
    for i in range(len(X)):
        shift= int(rng.uniform(-max_shift,max_shift))
        X[i]["x"]= X[i]["x"]+shift
        if shift > 0:
            idx= X[i]["x"] < 700*p["RESCALE_X"]
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
def ID_jitter(X,rng,sigma,p):
    for i in range(len(X)):
        shift= np.round(rng.standard_normal(len(X[i]["x"]))*sigma).astype(int)
        X[i]["x"]= X[i]["x"]+shift
        idx= np.logical_and(X[i]["x"] < 700*p["RESCALE_X"], X[i]["x"] >= 0)
        X[i]["x"]= X[i]["x"][idx]
        X[i]["t"]= X[i]["t"][idx]
    return X
    

"""
dilate or compress time
"""
def random_dilate(X,rng,min_factor,max_factor,p):
    mn= np.log(min_factor)
    mx= np.log(max_factor)
    for i in range(len(X)):
        fac= np.exp(rng.uniform(mn,mx))
        X[i]["t"]= X[i]["t"]*fac
        idx= X[i]["t"] < 1.4
        X[i]["t"]= X[i]["t"][idx]
        X[i]["x"]= X[i]["x"][idx]
    return X


"""
class for counting through bespoke tuples
"""

class incr:
    def __init__(self, ranges, fix= [], fix_v= []):
        self.x= np.zeros(len(ranges), dtype= int)
        self.rngs= ranges
        fix= np.array(fix, dtype= int)
        self.fix= fix[fix < len(ranges)]
        fix_v= np.array(fix_v)
        self.x[self.fix]= fix_v[fix < len(ranges)]
        self.calc_val()

    def calc_val(self):
        self.v= 0
        for i in range(len(self.rngs)):
            self.v= self.v*self.rngs[i]+self.x[i]
            
    def val(self):
        return self.v

    def next(self):
        if self.v is not None:
            cont= self.add(len(self.rngs)-1)
            if cont:
                self.calc_val()
            else:
                self.v= None
            
    def add(self,level):
        down= False
        if level not in self.fix:
            self.x[level]+= 1
            if self.x[level] >= self.rngs[level]:
                self.x[level]= 0
                down= True
        else:
            down= True
        if down:
            if level > 0:
                return self.add(level-1)
            else:
                return False
        else:
            return True
