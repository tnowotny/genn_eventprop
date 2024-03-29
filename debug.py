from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["DT_MS"]= 0.1
p["N_BATCH"]= 32
p["N_TRAIN"]= p["N_BATCH"]
p["N_EPOCH"]= 1
p["REC_SPIKES"]= ["input","hidden","output"]
p["REC_NEURONS"]= [("hidden","V"),
                   ("hidden", "lambda_V"),
                   ("hidden", "lambda_I"),
                   ("output", "V"),
                   ("output", "lambda_V"),
                   ("output", "lambda_I")
                   ]
p["REC_SYNAPSES"]= [("in_to_hid", "w"),
                    ("in_to_hid", "dw"),
                    ("hid_to_out", "w"),
                    ("hid_to_out", "dw")
                    ]
                    
spike_t, spike_ID, rec_vars_n, rec_vars_s= run_yinyang(p)

plt.figure()
plt.scatter(spike_t["input"], spike_ID["input"],s=0.5)
plt.scatter(spike_t["hidden"], spike_ID["hidden"]+6.0,s=0.5)
plt.scatter(spike_t["output"], spike_ID["output"]+207,s=0.5)
plt.figure()
plt.title("Vhidden")
plt.plot(rec_vars_n["Vhidden"])
plt.figure()
plt.title("lambda_Vhidden")
plt.plot(rec_vars_n["lambda_Vhidden"])
plt.figure()
plt.title("lambda_Ihidden")
plt.plot(rec_vars_n["lambda_Ihidden"])
plt.figure()
plt.title("Voutput")
plt.plot(rec_vars_n["Voutput"])
plt.figure()
plt.title("lambda_Voutput")
plt.plot(rec_vars_n["lambda_Voutput"])
plt.figure()
plt.title("lambda_Ioutput")
plt.plot(rec_vars_n["lambda_Ioutput"])
plt.figure()
plt.title("win_to_hid")
plt.plot(rec_vars_s["win_to_hid"])
plt.figure()
plt.title("dwin_to_hid")
plt.plot(rec_vars_s["dwin_to_hid"])
plt.figure()
print(rec_vars_s["whid_to_out"].shape)
plt.title("whid_to_out")
plt.plot(rec_vars_s["whid_to_out"][:,0:10])
plt.figure()
plt.title("whid_to_out_sum")
plt.plot(np.mean(rec_vars_s["whid_to_out"][:,0:-2:3],axis=1))
plt.plot(np.mean(rec_vars_s["whid_to_out"][:,1:-1:3],axis=1))
plt.plot(np.mean(rec_vars_s["whid_to_out"][:,2::3],axis=1))
plt.figure()
print(rec_vars_s["dwhid_to_out"].shape)
plt.title("dwhid_to_out")
plt.plot(rec_vars_s["dwhid_to_out"][:,0:10])
plt.figure()
plt.title("dwhid_to_out_sum")
plt.plot(np.mean(rec_vars_s["dwhid_to_out"][:,0:-2:3],axis=1))
plt.plot(np.mean(rec_vars_s["dwhid_to_out"][:,1:-1:3],axis=1))
plt.plot(np.mean(rec_vars_s["dwhid_to_out"][:,2::3],axis=1))

plt.show()
