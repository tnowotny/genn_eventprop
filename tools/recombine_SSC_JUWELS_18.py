import numpy as np
import json
import os

bname = "scan_SSC_JUWELS_18{}/JSSC18{}_scan_"
destination = "scan_SSC_final"

dt_ms= [ 1, 2, 5, 10, 20 ]
num_hidden= [ 64, 128, 256, 512, 1024 ]
sizes1 = [ 256, 1024]
sizes2 = [ 64, 128, 512 ]
delay = [ 0, 10 ]
shift = [ 0.0, 40.0 ]
blend = [ [], [0.5, 0.5] ]
train_tau = [False, True]
hid_neuron = ["LIF", "hetLIF"]

files = [
    "_best.txt",
    ".json",
    "_results.txt",
    "_test.json",
    "_test_results.txt",
    "_w_hidden0_hidden0_best.npy",
    "_w_hidden_output_best.npy",
    "_w_input_hidden_best.npy",
]

the_id = 0
for i in range(5):
    for j in range(5):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        for o in range(2):
                            for q in range(5):
                                for r in range(2):
                                    if (i < 2) or (num_hidden[j] not in sizes2):
                                        if (k == 1 and l == 1 and m == 1):
                                            if num_hidden[j] in sizes1:
                                                the_j = np.where(np.asarray(sizes1) == num_hidden[j])[0][0]
                                                id1 = (((((((i*2+the_j)*2+k)*2+l)*2+m)*2+n)*2+o)*5+q)*2+r
                                                basename = bname.format("","")
                                            else:
                                                the_j= np.where(np.asarray(sizes2) == num_hidden[j])[0][0]
                                                id1 = ((((i*3+the_j)*2+n)*2+o)*5+q)*2+r
                                                basename = bname.format("b","b")
                                        else:
                                            if num_hidden[j] in sizes1:
                                                the_j = np.where(np.asarray(sizes1) == num_hidden[j])[0][0]
                                                id1 = (((((((i*2+the_j)*2+k)*2+l)*2+m)*2+n)*2+o)*5+q)*2+r
                                                basename = bname.format("","")
                                            else:
                                                the_j= np.where(np.asarray(sizes2) == num_hidden[j])[0][0]
                                                id1 = (((((((i*3+the_j)*2+k)*2+l)*2+m)*2+n)*2+o)*5+q)*2+r
                                                basename = bname.format("c","c")
                                        filename = basename+str(id1)
                                        with open(filename+".json","r") as f:
                                            p= json.load(f)
                                        assert(p["DT_MS"] == dt_ms[i])
                                        assert(p["NUM_HIDDEN"] == num_hidden[j])
                                        assert(p["N_INPUT_DELAY"] == delay[k])
                                        if l == 1:
                                            assert(p["AUGMENTATION"]["random_shift"] == shift[l])
                                        if m == 1:
                                            assert(p["AUGMENTATION"]["blend"] == blend[m])
                                        assert(p["HIDDEN_NEURON_TYPE"] == hid_neuron[n])
                                        assert(p["TRAIN_TAU"] == train_tau[o])
                                        newname= f"SSC_{str(the_id).zfill(4)}"
                                        for f in files:
                                            cmd = f"cp {basename}{id1}{f} {destination}/{newname}{f}"
                                            #print(cmd)
                                            os.system(cmd)
                                        the_id += 1
