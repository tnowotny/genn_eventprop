from simulator import *
import matplotlib.pyplot as plt
import numpy as np

p["N_TRAIN"]= 10*p["N_BATCH"]
p["REC_SPIKES"]= ["input"]

spike_t, spike_ID= run_yingyang(p)

plt.figure()
plt.scatter(spike_t["input"], spike_ID["input"])
plt.show()


