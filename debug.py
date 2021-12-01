from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["N_TRAIN"]= 10*p["N_BATCH"]
p["REC_SPIKES"]= ["input","hidden","output"]
p["REC_NEURONS"]= [("hidden", "lambda_V"),
                   ("hidden", "lambda_I"),
                   ("output", "lambda_V"),
                   ("output", "lambda_I")
                   ]
p["REC_SYNAPSES"]= [("in_to_hid", "w"),
                    ("in_to_hid", "dw"),
                    ("hid_to_out", "w"),
                    ("hid_to_out", "dw")
                    ]
                    
spike_t, spike_ID, rec_vars_n, rec_vars_s= run_yingyang(p)

plt.figure()
plt.scatter(spike_t["input"], spike_ID["input"])
plt.scatter(spike_t["hidden"], spike_ID["hidden"]+6,"r")
plt.scatter(spike_t["output"], spike_ID["output"]+207,"g")
print(rec_vars_n)
plt.figure()
plt.plot(rec_vars_n["lambda_Vhidden"])
plt.figure()
plt.plot(rec_vars_n["lambda_Ihidden"])
plt.figure()
plt.plot(rec_vars_n["lambda_Voutput"])
plt.figure()
plt.plot(rec_vars_n["lambda_Ioutput"])
plt.figure()
plt.plot(rec_vars_s["win_to_hid"])
plt.figure()
plt.plot(rec_vars_s["dwin_to_hid"])
plt.figure()
plt.plot(rec_vars_s["whid_to_out"])
plt.figure()
plt.plot(rec_vars_s["dwhid_to_out"])

plt.show()
