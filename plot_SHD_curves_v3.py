import numpy as np
import matplotlib.pyplot as plt
import json
import sys

if len(sys.argv) < 2:
    print("usage: python plot_SHD_curves_v3.py XXX.json (summary of get_best**)")
    exit(1)

fname= sys.argv[1]
with open(fname,"r") as f:
    best= json.load(f)

ptrain= []
ltrain= []
ptest= []
ltest= []

N= 10
N_epoch= 300

for b in best:
    with open(b["path2"]+str(b["id2"])+"_results.txt", "r") as f:
        d= np.loadtxt(f)
        print(d.shape)
        d= d[:N_epoch*N,:]
        ptrain.append(d[:,1].reshape((N,N_epoch)).T)
        ltrain.append(d[:,2].reshape((N,N_epoch)).T)
        ptest.append(d[:,3].reshape((N,N_epoch)).T)
        ltest.append(d[:,4].reshape((N,N_epoch)).T)

idx= [ 0, 4, 1, 5, 2, 6, 3, 7 ]

ptrain= [ ptrain[idx[i]] for i in range(8) ]
ptest= [ ptest[idx[i]] for i in range(8) ]
ltrain= [ ltrain[idx[i]] for i in range(8) ]
ltest= [ ltest[idx[i]] for i in range(8) ]

plt.figure(figsize= (8,4.5))
for p in ptrain:
    m= np.mean(p,axis=1)
    plt.plot(m)
    s= np.std(p,axis=1)
    plt.fill_between(range(N_epoch), m-s, m+s,alpha=0.4)
    plt.xlim([ 0, 400 ])
    plt.legend(["sum of V \n (ffwd)", "sum of V \n (recur)", "sum_exp \n of V (ffwd)", "sum_exp \n of V (recur)", "time of first \n spike (ffwd)", "time of first \n spike (recur)", "max of V \n (ffwd)","max of V \n (recur)" ])
    plt.ylim([ 0, 1.05 ])
    plt.title("training performance")
    plt.xlabel("epoch")
    plt.ylabel("fraction correct")
    plt.savefig("best_training_curve.png",dpi=300)
    
plt.figure(figsize= (6,4.5))
for p in ptest:
    m= np.mean(p,axis=1)
    plt.plot(m)
    s= np.std(p,axis=1)
    plt.fill_between(range(N_epoch), m-s, m+s,alpha=0.4)    
    plt.ylim([ 0, 1.05 ])
    plt.title("testing performance")
    plt.xlabel("epoch")
    plt.ylabel("fraction correct")
    plt.savefig("best_testing_curve.png",dpi=300)

plt.figure()
for p in ltrain:
    plt.plot(np.mean(p,axis=1))
    plt.legend(["sum of V \n (ffwd)", "sum of V \n (recur)", "sum_exp \n of V (ffwd)", "sum_exp \n of V (recur)", "time of first \n spike (ffwd)", "time of first \n spike (recur)", "max of V \n (ffwd)","max of V \n (recur)" ])
    plt.title("training loss")

plt.figure()
for p in ltest:
    plt.plot(np.mean(p,axis=1))
    plt.legend(["sum of V \n (ffwd)", "sum of V \n (recur)", "sum_exp \n of V (ffwd)", "sum_exp \n of V (recur)", "time of first \n spike (ffwd)", "time of first \n spike (recur)", "max of V \n (ffwd)","max of V \n (recur)" ])
    plt.title("testing loss")

plt.show()
