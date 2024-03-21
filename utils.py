import numpy as np
import copy
from dataclasses import dataclass
from typing import Tuple
import json

"""
this is an augmentation for SHD spoken digits
spiking data is shifted by a random number of neurons along the neuron ID axis
"""
def random_shift(X,rng,max_shift,num_input):
    for i in range(len(X)):
        shift= int(rng.uniform(-max_shift,max_shift))
        X[i]["x"]= X[i]["x"]+shift
        if shift > 0:
            idx= X[i]["x"] < num_input
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
def ID_jitter(X,rng,sigma,num_input):
    for i in range(len(X)):
        shift= np.round(rng.standard_normal(len(X[i]["x"]))*sigma).astype(int)
        X[i]["x"]= X[i]["x"]+shift
        idx= np.logical_and(X[i]["x"] < num_input, X[i]["x"] >= 0)
        X[i]["x"]= X[i]["x"][idx]
        X[i]["t"]= X[i]["t"][idx]
    return X
    

"""
dilate or compress time
"""
def random_dilate(X,rng,min_factor,max_factor,trial_ms):
    mn= np.log(min_factor)
    mx= np.log(max_factor)
    for i in range(len(X)):
        fac= np.exp(rng.uniform(mn,mx))
        X[i]["t"]= X[i]["t"]*fac
        idx= X[i]["t"] < trial_ms
        X[i]["t"]= X[i]["t"][idx]
        X[i]["x"]= X[i]["x"][idx]
    return X


"""
blend spike patterns
WARNING: This does not ensure one spike per timestep!
"""
def blend(X,probs,rng,num_input,trial_ms):
    new_x= []
    new_t= []
    X= copy.deepcopy(X)     # ARGH - need to be extremely careful not to fuck up orig data!
    mx= np.zeros(len(X))
    mt= np.zeros(len(X))
    for i,d in enumerate(X):
        mx[i]= np.mean(d["x"])
        mt[i]= np.mean(d["t"])
    mmx= np.mean(mx)
    mmt= np.mean(mt)
    for i in range(len(X)):
        X[i]["x"]+= int(mmx-mx[i])
        X[i]["t"]+= int(mmt-mt[i])
    for i,d in enumerate(X):        
        pp= rng.uniform(0,1,len(d["t"]))
        new_x.extend(d["x"][pp < probs[i]])
        new_t.extend(d["t"][pp < probs[i]])
    idx= np.argsort(new_t)
    new_t= np.array(new_t)[idx]
    new_x= np.array(new_x)[idx]
    keep= np.logical_and(new_t >= 0, new_t < trial_ms)
    new_t= new_t[keep]
    new_x= new_x[keep]
    keep= np.logical_and(new_x >= 0, new_x < num_input)
    new_t= new_t[keep]
    new_x= new_x[keep]
    assert(max(new_x) < 700)
    return {"t": new_t, "x": new_x} 

"""
extend a dataset with blended spike patterns
"""

def blend_dataset(X, Y, Z, rng, probs, n_train, num_input, trial_ms):
    nblend= n_train-len(X)    # number of blended examples to generate
    new_X= []
    new_Y= []
    if Z is not None:
        new_Z= []
    for i in range(nblend):
        idx= rng.integers(0,len(X))
        #td= X[np.logical_and(Y == Y[idx], Z == Z[idx])]
        td= X[Y == Y[idx]]
        td= td[rng.integers(0,len(td),len(probs))]
        new_X.append(blend(td,probs,rng,num_input,trial_ms))
        new_Y.append(Y[idx])
        if Z is not None:
            new_Z.append(Z[idx])
    X= np.hstack([X,new_X])
    Y= np.hstack([Y,new_Y])
    if Z is not None:
        Z= np.hstack([Z,new_Z])
    return X, Y, Z
               
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


@dataclass
class EventsToGrid:
    sensor_size: Tuple[int, int, int]
    dt: float

    def __call__(self, events):
        # Tuple of possible axis names
        axes = ("x", "y", "p")

        # Build bin and sample data structures for histogramdd
        bins = []
        sample = []
        for s, a in zip(self.sensor_size, axes):
            if a in events.dtype.names:
                bins.append(np.linspace(0, s, s + 1))
                sample.append(events[a])

        # Add time bins
        bins.append(np.arange(0.0, np.amax(events["t"]) + self.dt, self.dt))
        sample.append(events["t"])

        # Build histogram
        event_hist,_ = np.histogramdd(tuple(sample), tuple(bins))
        new_events = np.where(event_hist > 0)

        # Copy x, y, p data into new structured array
        grid_events = np.empty(len(new_events[0]), dtype=events.dtype)
        for i, a in enumerate(axes):
            if a in grid_events.dtype.names:
                grid_events[a] = new_events[i]

        # Add t, scaling by dt
        grid_events["t"] = new_events[-1] * self.dt
        return grid_events
       

def gridlines(ax, hti, wdi, split):
    k= 1
    ymn= 0
    ymx= np.product(split[:hti])
    xmn= 0
    xmx= np.product(split[hti:])
    for i in range(hti):
        k*= split[i]
        d= ymx // k
        lnsy= [ [j*d-0.5, j*d-0.5] for j in range(1,ymx // d) ]
        lnsx= [ [xmn-0.5, xmx-0.5] for j in range(1,ymx // d) ]
        ax.plot(np.array(lnsx).T,np.array(lnsy).T,'w',lw= hti-i)   
    k=1
    for j in range(hti, len(split)):
        k*= split[j]
        d= xmx // k
        lnsx= [ [i*d-0.5, i*d-0.5] for i in range(1,xmx // d) ]
        lnsy= [ [ymn-0.5, ymx-0.5] for i in range(1,xmx // d) ]
        ax.plot(np.array(lnsx).T,np.array(lnsy).T,'w',lw= len(split)-j)

def remap(d, split, remap):
    for j in range(d.shape[0]):
        x= d[j,:].copy()
        x= np.reshape(x, split)
        x= x.transpose(remap)
        d[j,:]= x.flatten()
    split= np.asarray(split)[remap]
    return d, split

def average_out(d, s, split, avg, names):
    avg= -np.sort(-np.asarray(avg)) # sort descending so that the pop() works with consistent indexing
    d2= []
    for j in range(d.shape[0]):
        x= np.reshape(d[j,:].copy(), split)
        for i in avg:
            x= np.mean(x,axis= i)
        d2.append(x.flatten())
    d2 = np.array(d2) 
    nsplit= split.copy()
    nnames= names.copy()
    for x in avg:
        nsplit.pop(x)
        nnames.pop(x)
        s= s // split[x]
    return (d2, s, nsplit, nnames)

def optimise(d, lst, opt, split):
    split= np.asarray(split)
    d2= d.copy().reshape(split)
    d2 = np.transpose(d2, lst+opt)
    ny = np.prod(split[lst])
    nx = np.prod(split[opt])
    d2 = d2.reshape((ny,nx))
    best_idx = np.argmax(d2, axis=1)
    best = [ i*nx+best_idx[i] for i in range(ny) ]
    return best


def load_train_test(basename,s,N_avg,secondary,reserve=None):
    res_col = 12
    results = [ [] for i in range(res_col) ] # 11 types of results
    for i in range(s):
        d2swap = False
        fname= basename+"_"+str(i)+".json"
        results[0].append(i)
        try:
            with open(fname,"r") as f:
                p= json.load(f)
        except:
            for j in range(res_col-1):
                results[j+1].append(0)        
                print(f"error trying to load {fname}")
        else:
            fname= basename+"_"+str(i)+"_results.txt"
            try:
                with open(fname, "r") as f:
                    d= np.loadtxt(f)
            except:
                for j in range(0,5):
                    results[j+1].append(0)        
                results[11].append(0)
                print("error trying to load {}".format(fname))
            else:
                if reserve is not None:
                    fname = reserve+"_"+str(i)+"_results.txt"
                    try:
                        with open(fname, "r") as f:
                            d2= np.loadtxt(f)
                    except:
                        print("no reserve")
                    else:
                        # if d2 is a better result in terms of final training error
                        if (d2[-1,1] > d[-1,1]):
                            d= d2
                            d2swap = True
                results[1].append(len(d[:,1]))
                for j in range(1,5):
                    results[j+1].append(np.mean(d[-N_avg:,j]))
                results[11].append(d[-1,-1]/(d[-1,0]+1))
            if d2swap:
                fname = reserve+"_"+str(i)+"_"+secondary+".txt"
            else:
                fname = basename+"_"+str(i)+"_"+secondary+".txt"
            try:
                with open(fname, "r") as f:
                    d= np.loadtxt(f)
            except:
                for j in range(0,5):
                    results[j+6].append(0)
                print("error trying to load {}".format(fname))
            else:
                results[6].append(d[0])
                for j in range(1,5):
                    results[j+6].append(d[j])
    for x in results:
        print(len(x))
    results= np.array(results)
    return results


# some tools for looking at parameter dictionaries

def print_diff(p, p1):
    for x in p:
        if x not in p1:
            print(f"{x} not in p1")
        else:
            if p[x] != p1[x]:
                print(f"{x}: {p[x]} - {p1[x]}")
    for x in p1:
        if x not in p:
            print(f"{x} not in p")
    print("--------------------------------------------------")
            

def diff_keys(p, p1):
    # only considering common keys that differ in value
    # print a warning if a key doen't exist in one of them
    d= set()
    for x in p:
        if x not in p1:
            print(f"{x} not in p1")
        else:
            if p[x] != p1[x]:
                d.add(x)
    for x in p1:
        if x not in p:
            print(f"{x} not in p")
    return d

def subdict(p : dict, k : set):
    pnew= {}
    for x in k:
        pnew[x]= p[x]
    return pnew
