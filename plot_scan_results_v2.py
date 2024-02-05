
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

def gridlines(ax, hti, wdi, split):
    k= 1
    ymn= 0
    ymx= np.product(split[:hti])
    xmn= 0
    xmx= np.product(split[hti:])
    for i in range(hti):
        k*= split[i]
        d= ymx // k
        lnsy= [ [j*d-0.5, j*d-0.5] for j in range(ymx // d) ]
        lnsx= [ [xmn-0.5, xmx-0.5] for j in range(ymx // d) ]
        ax.plot(np.array(lnsx).T,np.array(lnsy).T,'w',lw= hti-i)   
    k=1
    for j in range(hti, len(split)):
        k*= split[j]
        d= xmx // k
        lnsx= [ [i*d-0.5, i*d-0.5] for i in range(xmx // d) ]
        lnsy= [ [ymn-0.5, ymx-0.5] for i in range(xmx // d) ]
        ax.plot(np.array(lnsx).T,np.array(lnsy).T,'w',lw= len(split)-j)

def remap(d, ht, wd, split, remap):
    x= np.reshape(d, split)
    x= x.transpose(remap)
    x= x.reshape(ht,wd)
    return x

def average_out(d, hti, ht, wd, split, avg):
    x= np.reshape(d, split)
    avg.sort(reverse= True)
    s= split.copy()
    for i in avg:
        if i < hti:
            ht= ht // s[i]
        else:
            wd= wd // s[i]
        s.pop(i)
        x= np.mean(x,axis= i)
    x= x.reshape(ht,wd)
    return (x, ht, wd, s)

if len(sys.argv) != 3:
    print("Usage: python plot_scan_results.py <basename> <settings.json>")
    exit(1)

basename= sys.argv[1]

with open(sys.argv[2],"r") as f:
    para= json.load(f)

# decide on x/y split
s= np.product(para["split"])
i= 0
ht= 1
while ht < np.sqrt(s):
    ht*= para["split"][i]
    i+= 1

i-= 1
ht= ht // para["split"][i]
hti= i
wd= int(np.product(para["split"][hti:]))
wdi= len(para["split"])-hti
    
N_S= para["epoch"] # show results at this epoch

mn= 10.0
N_avg= 5
N_spk= 4
final_cor= np.zeros((ht,wd))
final_cor_e= np.zeros((ht,wd))
nepoch= np.zeros((ht,wd))
#fig, ax= plt.subplots(ht,wd,sharex= True, sharey= True)
for i in range(ht):
    for j in range(wd):
        nn= i*wd+j
        fname= basename+"_"+str(nn)+".json"
        try:
            with open(fname,"r") as f:
                p= json.load(f)
        except:
            print(f"error trying to load {fname}")
        else:
            N_E= p["N_EPOCH"] # total number of epochs for each speaker
            idx= np.outer(np.arange(N_S-N_avg,N_spk*N_E,N_E),np.ones(N_avg))+np.outer(np.ones(N_spk),np.arange(N_avg))
            idx= np.array(idx.flatten(), dtype= int)

            fname= basename+"_"+str(nn)+"_results.txt"
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
                    nepoch[i,j]= len(d[:,1])
                    try:
                        final_cor[i,j]= np.mean(d[idx,1])
                        final_cor_e[i,j]= np.mean(d[idx,3])
                    except:
                        final_cor[i,j]= 0
                        final_cor_e[i,j]= 0
                        print(f"error taking the mean, data length {d.shape[0]}")
#            ax[i,j].set_title("scan_"+str(i*wd+j),fontsize= 6)
#print(mn)
#plt.yscale("log")
#fig.savefig(basename+"_accuracy.png")

#fig, ax= plt.subplots(1,2,sharey=True)
#for i in range(ht):
#    for j in range(wd):
#        fname= basename+"_"+str(i*wd+j)+"_results.txt"
#        try:
#            with open(fname, "r") as f:
#                d= np.loadtxt(f)
#        except:
#            print("error trying to load {}".format(fname))
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
#print(mn)
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
print(final_loss)
print(final_loss_e)
print(nepoch)
mx= np.max(np.max(final_cor_e))
i, j= np.where(final_cor_e == mx)
print(f"{i},{j}: eval {mx}")

if "avg" in para:
    nepoch, nht, nwd, split= average_out(nepoch, hti, ht, wd, para["split"], para["avg"]) 
    final_cor, nht, nwd, split= average_out(final_cor, hti, ht, wd, para["split"], para["avg"])
    final_cor_e, nht, nwd, split= average_out(final_cor_e, hti, ht, wd, para["split"], para["avg"])
    final_loss, nht, nwd, split= average_out(final_loss, hti, ht, wd, para["split"], para["avg"])
    final_loss_e, nht, nwd, split= average_out(final_loss_e, hti, ht, wd, para["split"], para["avg"])
    para["split"]= split
    ht= nht
    wd= nwd
    
if "remap" in para:
    nepoch= remap(nepoch, ht, wd, para["split"], para["remap"]) 
    final_cor= remap(final_cor, ht, wd, para["split"], para["remap"])
    final_cor_e= remap(final_cor_e, ht, wd, para["split"], para["remap"])
    final_loss= remap(final_loss, ht, wd, para["split"], para["remap"])
    final_loss_e= remap(final_loss_e, ht, wd, para["split"], para["remap"])
    
plt.figure()
plt.imshow(nepoch,interpolation='none',cmap='jet')
plt.colorbar()
fig, ax= plt.subplots(2,2)
im= ax[0,0].imshow(final_cor,interpolation='none',vmin= 0.7, vmax= 1.0, cmap='jet')
fig.colorbar(im,ax=ax[0,0])
gridlines(ax[0,0],hti, wdi, para["split"])
ax[0,0].set_title(f"training correct e={N_S}")
im= ax[0,1].imshow(final_cor_e,interpolation='none',vmin= 0.85, vmax= 0.9, cmap='jet')
fig.colorbar(im,ax=ax[0,1])
gridlines(ax[0,1],hti, wdi, para["split"])
ax[0,1].set_title(f"evaluation correct e={N_S}")
im= ax[1,0].imshow(final_loss,interpolation='none', cmap='jet')
fig.colorbar(im,ax=ax[1,0])
gridlines(ax[1,0],hti, wdi, para["split"])
ax[1,0].set_title(f"training loss e={N_S}")
im= ax[1,1].imshow(final_loss_e,interpolation='none', cmap='jet')
fig.colorbar(im,ax=ax[1,1])
gridlines(ax[1,1],hti, wdi, para["split"])
ax[1,1].set_title(f"evaluation loss e={N_S}")
fig.savefig(basename+"summary.png",dpi=300)
plt.show()
