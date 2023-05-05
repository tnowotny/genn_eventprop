
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

if len(sys.argv) != 5:
    print("Usage: python plot_scan_results.py <basename> <ht> <wd> <epoch>")
    exit(1)

basename= sys.argv[1]
ht= int(sys.argv[2])
wd= int(sys.argv[3])

fname= basename+"_6.json"
with open(fname,"r") as f:
    p= json.load(f)

N_E= p["N_EPOCH"] # total number of epochs for each speaker
N_S= int(sys.argv[4]) # show results at this epoch

mn= 10.0
idx= np.outer(np.arange(N_S-10,10*N_E,N_E),np.ones(10))+np.outer(np.ones(10),np.arange(10))
print(idx)
idx= np.array(idx.flatten(), dtype= int)
final_cor= np.zeros((ht,wd))
final_cor_e= np.zeros((ht,wd))
#fig, ax= plt.subplots(ht,wd,sharex= True, sharey= True)
for i in range(ht):
    for j in range(wd):
        fname= basename+"_"+str(i*wd+j)+"_results.txt"
        try:
            with open(fname, "r") as f:
                d= np.loadtxt(f)
        except:
            print("error trying to load {}".format(fname))
        else:
            if len(d) > 0:
#                ax[i,j].plot(1-d[:,1])
#                ax[i,j].plot(1-d[:,3])
                mn= np.min([mn,np.amin(1-d[:,3])])
                try:
                    final_cor[i,j]= np.mean(d[idx,1])
                    final_cor_e[i,j]= np.mean(d[idx,3])
                except:
                    final_cor[i,j]= 0
                    final_cor_e[i,j]= 0
#            ax[i,j].set_title("scan_"+str(i*wd+j),fontsize= 6)
print(mn)
#plt.yscale("log")
#fig.savefig(basename+"_accuracy.png")

#fig, ax= plt.subplots(1,2,sharey=True)
for i in range(ht):
    for j in range(wd):
        fname= basename+"_"+str(i*wd+j)+"_results.txt"
        try:
            with open(fname, "r") as f:
                d= np.loadtxt(f)
        except:
            print("error trying to load {}".format(fname))
#        else:
#            if len(d) > 0:
#                ax[0].plot(1-d[:,1])
#                ax[1].plot(1-d[:,3])
#plt.yscale("log")
#fig.savefig(basename+"_accuracy_2.png")

final_loss= np.zeros((ht,wd))
final_loss_e= np.zeros((ht,wd))
#fig, ax= plt.subplots(ht,wd,sharex= True, sharey= True)
mn= 10.0
for i in range(ht):
    for j in range(wd):
        fname= basename+"_"+str(i*wd+j)+"_results.txt"
        try:
            with open(fname, "r") as f:
                d= np.loadtxt(f)
        except:
            print("error trying to load {}".format(fname))
        else:
            if len(d) > 0:
#                ax[i,j].plot(d[:,2])
#                ax[i,j].plot(d[:,4])
                mn= np.min([mn,np.amin(d[:,4])])
                try:
                    final_loss[i,j]= np.mean(d[idx,2])
                    final_loss_e[i,j]= np.mean(d[idx,4])
                except:
                    final_loss[i,j]= 0
                    final_loss_e[i,j]= 0
#            ax[i,j].set_title("scan_"+str(i*wd+j))
#plt.yscale("log")
#plt.ylim([0.01,10])
print(mn)
#fig.savefig(basename+"_loss.png")

"""
fig, ax= plt.subplots(ht,wd,sharex= True, sharey= True)
for i in range(ht):
    for j in range(wd):
        fname= basename+"_"+str(i*wd+j)+"_results.txt"
        try:
            with open(fname, "r") as f:
                d= np.loadtxt(f)
        except:
            print("error trying to load {}".format(fname))
        else:
            if len(d) > 0:
                ax[i,j].errorbar(np.arange(len(d)),d[:,5],yerr=d[:,6])
                ax[i,j].plot(d[:,7])
                ax[i,j].plot(d[:,8])
            ax[i,j].set_title("scan_"+str(i*wd+j))
fig.savefig(basename+"_activity.png")
"""

print(final_cor)
print(final_cor_e)
mx= np.max(np.max(final_cor_e))
i, j= np.where(final_cor_e == mx)
print(i,j)
print(final_loss)
print(final_loss_e)

fig, ax= plt.subplots(2,2)
im= ax[0,0].imshow(final_cor,interpolation='none',vmin= 0.7, vmax= 1.0, cmap='jet')
fig.colorbar(im,ax=ax[0,0])
im= ax[0,1].imshow(final_cor_e,interpolation='none',cmap='jet')
fig.colorbar(im,ax=ax[0,1])
im= ax[1,0].imshow(final_loss,interpolation='none', cmap='jet')
fig.colorbar(im,ax=ax[1,0])
im= ax[1,1].imshow(final_loss_e,interpolation='none', cmap='jet')
fig.colorbar(im,ax=ax[1,1])

plt.show()
