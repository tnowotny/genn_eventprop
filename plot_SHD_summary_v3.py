import numpy as np
import matplotlib.pyplot as plt
import json
import sys

if len(sys.argv) < 2:
    print("usage: python plot_SHD_summary_v3.py XXX.json (summary of get_best**)")
    exit(1)

fname= sys.argv[1]
with open(fname,"r") as f:
    x= json.load(f)

extra= [[ 0, 0, 0.89431094, 0.655771495877503],
	[ 0, 0, 0.93746935, 0.7881625441696113],
	[ 0, 0, 0.93746935, 0.8]]

extraS= [[ 0, 0, 0, 0.018468086315770287],
	[ 0, 0, 0, 0.010573433703435116],
	[ 0, 0, 0, 0.02 ]] 

ar= []
for res in x:
    r= res["perf"]
    ar.append(r)

ar= np.array(ar)

# separate mean and std
totmn= ar[:,::2]
totstd= ar[:,1::2]


# do two separate figures with and without augmentation
for a in range(2):
    
    # add the values from the literature 
    xmn= np.vstack([extra,totmn[8*a:8*(a+1)]])
    xstd= np.vstack([extraS,totstd[8*a:8*(a+1)]])
   
    plt.figure(figsize=[ 10, 5 ])
    plt.grid(visible=True, which= 'major', axis= 'y', color='k', linestyle=':', linewidth= 0.5)
    for m in range(4):
        plt.bar(np.arange(11)+0.2*m, xmn[:,m],width=0.18)
    for m in range(4):
        plt.errorbar(np.arange(11)+0.2*m, xmn[:,m], xstd[:,m],ls='none',color='k')
    plt.ylim([ 0, 1.1])
    plt.xticks(np.arange(11)+0.3)
    plt.gca().set_xticklabels(["Eprop with \n LIF (recur)", "Eprop with \n ALIF (recur)", "BPTT with  \n surrogate gra- \n dient (recur)", "sum of V \n (ffwd)", "sum of V \n (recur)", "sum_exp \n of V (ffwd)", "sum_exp \n of V (recur)", "time of first \n spike (ffwd)", "time of first \n spike (recur)", "max of V \n (ffwd)","max of V \n (recur)" ] , rotation= 60)
    plt.ylabel("fraction correct")
    plt.legend(["training xval", "validation xval", "training", "test"],loc="upper right")
    plt.tight_layout()
    plt.savefig("SHD_all_overview_"+str(a)+".png", dpi=300)
    plt.show()

