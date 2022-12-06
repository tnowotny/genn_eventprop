import numpy as np
import matplotlib.pyplot as plt
import json
import sys


ptrainx= []
ltrainx= []
pval= []
lval= []
ptrain= []
ltrain= []
ptest= []
ltest= []

N= 10
N_epoch= 50
path= "test_MNIST_all/"
names= ["MNIST_avg_xentropy", "MNIST_sum", "MNIST_max", "MNIST_first_spike" ]
for n in names:
    with open(path+n+"_results.txt", "r") as f:
        d= np.loadtxt(f)
        print(d.shape)
        #d= d[:N_epoch*N,:]
        ptrainx.append(d[:N*N_epoch,1].reshape((N,N_epoch)).T)
        ltrainx.append(d[:N*N_epoch,2].reshape((N,N_epoch)).T)
        pval.append(d[:N*N_epoch,3].reshape((N,N_epoch)).T)
        lval.append(d[:N*N_epoch,4].reshape((N,N_epoch)).T)
        ptrain.append(d[N*N_epoch:,1].reshape((N,N_epoch)).T)
        ltrain.append(d[N*N_epoch:,2].reshape((N,N_epoch)).T)
        ptest.append(d[N*N_epoch:,3].reshape((N,N_epoch)).T)
        ltest.append(d[N*N_epoch:,4].reshape((N,N_epoch)).T)

plt.figure(figsize= (8,4.5))
for p in ptrainx:
    m= np.mean(p,axis=1)
    plt.plot(m)
plt.xlim([ 0, 66 ])
plt.ylim([0.82,1])
plt.legend(["average \ncross-entropy", "sum of V", "max of V", "time to \nfirst spike" ])
plt.title("training performance (cross-validation)")
plt.xlabel("epoch")
plt.ylabel("fraction correct")
for p in ptrainx:
    s= np.std(p,axis=1)
    plt.fill_between(range(N_epoch), m-s, m+s,alpha=0.4)
plt.savefig(path+"training_val_curves.png",dpi=300)
    
plt.figure(figsize= (6,4.5))
for p in pval:
    m= np.mean(p,axis=1)
    plt.plot(m)
    s= np.std(p,axis=1)
    plt.fill_between(range(N_epoch), m-s, m+s,alpha=0.4)    
plt.ylim([0.82,1])
plt.title("validation performance")
plt.xlabel("epoch")
plt.ylabel("fraction correct")
plt.savefig(path+"validation_curves.png",dpi=300)

plt.figure()
for p in ltrainx:
    plt.plot(np.mean(p,axis=1))
plt.legend(["average \ncross-entropy", "sum of V", "max of V", "time to \nfirst spike" ])
plt.title("training loss (cross-validation)")

plt.figure()
for p in lval:
    plt.plot(np.mean(p,axis=1))
plt.legend(["average \ncross-entropy", "sum of V", "max of V", "time to \nfirst spike" ])
plt.title("validation loss")

plt.figure(figsize= (8,4.5))
for p in ptrain:
    m= np.mean(p,axis=1)
    plt.plot(m)
plt.xlim([ 0, 66 ])
plt.ylim([0.82,1])
plt.legend(["average \ncross-entropy", "sum of V", "max of V", "time to \nfirst spike" ])
plt.title("training performance")
plt.xlabel("epoch")
plt.ylabel("fraction correct")
for p in ptrain:
    s= np.std(p,axis=1)
    plt.fill_between(range(N_epoch), m-s, m+s,alpha=0.4)
plt.savefig(path+"training_curves.png",dpi=300)
    
plt.figure(figsize= (6,4.5))
for p in ptest:
    m= np.mean(p,axis=1)
    plt.plot(m)
    s= np.std(p,axis=1)
    plt.fill_between(range(N_epoch), m-s, m+s,alpha=0.4)    
plt.ylim([0.82,1])
plt.title("testing performance")
plt.xlabel("epoch")
plt.ylabel("fraction correct")
plt.savefig(path+"testing_curves.png",dpi=300)
    
plt.figure()
for p in ltrain:
    plt.plot(np.mean(p,axis=1))
plt.legend(["average \ncross-entropy", "sum of V", "max of V", "time to \nfirst spike" ])
plt.title("training loss")

plt.figure()
for p in ltest:
    plt.plot(np.mean(p,axis=1))
plt.legend(["average \ncross-entropy", "sum of V", "max of V", "time to \nfirst spike" ])
plt.title("testing loss")

plt.show()
