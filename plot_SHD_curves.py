import numpy as np
import matplotlib.pyplot as plt

"""
i == 0, k == 0 - sum ffwd: scan_JADE_26
i == 0, k == 1 - sum recur: 
"""

#data_name= "scan_JADE_26/J26_scan_"
data_name= "scan_JADE_27/J27_scan_"
j= 1
l= 0
m= 0
n= 0
n2= 0
n3= 1
o= 0

ptrain= []
ltrain= []
ptest= []
ltest= []

N= 7
N_epoch= 300

for i in range(4):
    for k in range(2):
        scan_no= ((i*2+j)*2+k)*2+m
        if i == 3 or i == 0:
            scan_no+= 1 # consider the slightly stronger regularisation
        print(scan_no)
        jname= str(scan_no)+"_results.txt"
        d= np.loadtxt(data_name+jname)
        print(d.shape)
        d= d[:N_epoch*N,:]
        ptrain.append(d[:,1].reshape((N,N_epoch)).T)
        ltrain.append(d[:,2].reshape((N,N_epoch)).T)
        ptest.append(d[:,3].reshape((N,N_epoch)).T)
        ltest.append(d[:,4].reshape((N,N_epoch)).T)

#idx= [ 0, 4, 1, 5, 2, 6, 3, 7 ]

#print(perf)

#ptrain= [ ptrain[idx[i]] for i in range(8) ]
#ptest= [ ptest[idx[i]] for i in range(8) ]
#ltrain= [ ltrain[idx[i]] for i in range(8) ]
#ltest= [ ltest[idx[i]] for i in range(8) ]

plt.figure()
for p in ptrain:
    plt.plot(np.mean(p,axis=1))
    plt.legend(["sum of V \n (ffwd)", "sum of V \n (recur)", "sum_exp \n of V (ffwd)", "sum_exp \n of V (recur)", "time of first \n spike (ffwd)", "time of first \n spike (recur)", "max of V \n (ffwd)","max of V \n (recur)" ])
    plt.title("training performance")

plt.figure()
for p in ptest:
    plt.plot(np.mean(p,axis=1))
    plt.legend(["sum of V \n (ffwd)", "sum of V \n (recur)", "sum_exp \n of V (ffwd)", "sum_exp \n of V (recur)", "time of first \n spike (ffwd)", "time of first \n spike (recur)", "max of V \n (ffwd)","max of V \n (recur)" ])
    plt.title("testing preformance")

plt.figure()
for p in ltrain:
    plt.plot(np.mean(p,axis=1))
    plt.ylim([ 0, 1.05 ]
    plt.legend(["sum of V \n (ffwd)", "sum of V \n (recur)", "sum_exp \n of V (ffwd)", "sum_exp \n of V (recur)", "time of first \n spike (ffwd)", "time of first \n spike (recur)", "max of V \n (ffwd)","max of V \n (recur)" ])
    plt.title("training loss")

plt.figure()
for p in ltest:
    plt.plot(np.mean(p,axis=1))
    plt.ylim([ 0, 1.05 ]
    plt.legend(["sum of V \n (ffwd)", "sum of V \n (recur)", "sum_exp \n of V (ffwd)", "sum_exp \n of V (recur)", "time of first \n spike (ffwd)", "time of first \n spike (recur)", "max of V \n (ffwd)","max of V \n (recur)" ])
    plt.title("testing loss")

plt.show()
