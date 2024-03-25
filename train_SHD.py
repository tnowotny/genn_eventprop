from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import exists

p["EVALUATION"] = "speaker"
p["LOSS_TYPE"] = "first_spike_exp"
p["REG_TYPE"] = "simple"
p["LBD_UPPER"] = 1e-7
p["LBD_LOWER"] = 1e-7
p["INPUT_HIDDEN_MEAN"] = 0.03
p["PDROP_INPUT"] = 0.0
p["TAU_MEM"] = 40.0
p["AVG_SNSUM"] = True
jname= os.path.join(p["OUT_DIR"], p["NAME"]+".json")
jfile= open(jname,'w')
json.dump(p,jfile)
print(p)

mn= SHD_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

rname= os.path.join(p["OUT_DIR"], p["NAME"]+'.summary.txt')
sumfile= open(rname,'w')
sumfile.write("Training correct: {}, Valuation correct: {}".format(correct,correct_eval))

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
