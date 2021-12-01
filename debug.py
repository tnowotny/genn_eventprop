from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["N_TRAIN"]= 10*p["N_BATCH"]
p["REC_SPIKES"]= ["input"]
p["REC_NEURONS"]= {"hidden": "lambda_V",
                   "hidden": "lambda_I",
                   "output": "lambda_V",
                   "output": "lambda_I"
                   }
p["REC_SYNAPSES"]= {"in_to_hid": "w",
                    "in_to_hid": "dw",
                    "hid_to_out": "w",
                    "hid_to_out": "dw"
                    }
                    
spike_t, spike_ID, rec_vars_n, rec_vars_s= run_yingyang(p)

plt.figure()
plt.scatter(spike_t["input"], spike_ID["input"])

plt.figure()
plt.plot(rec_vars_n["hiddenlambdaV"])
plt.figure()
plt.plot(rec_vars_n["hiddenlambdaI"])
plt.figure()
plt.plot(rec_vars_n["outputlambdaV"])
plt.figure()
plt.plot(rec_vars_n["outputlambdaI"])
plt.figure()
plt.plot(rec_vars_n["in_to_hidw"])
plt.figure()
plt.plot(rec_vars_n["in_to_hiddw"])
plt.figure()
plt.plot(rec_vars_n["hid_to_outw"])
plt.figure()
plt.plot(rec_vars_n["hid_to_outdw"])

plt.show()



