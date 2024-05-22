import numpy as np
import json

def load_traintest_SHD_final(N_avg):

    dt_ms= [ 1, 2 ]
    num_hidden= [ 64, 128, 256, 512, 1024 ]
    delay = [ 0, 10 ]
    shift = [ 0.0, 40.0 ]
    blend = [ [], [0.5, 0.5] ]
    train_tau = [False, True]
    hid_neuron = ["LIF", "hetLIF"]

    res_col = 12
    results = [ [] for i in range(res_col) ] # 12 types of results
    split= [ 2, 5, 2, 2, 2, 2, 2, 8 ]
    s= np.prod(split)
    basename = "scan_SHD_final_traintest/SHD_tt"
    for i in range(2):
        for j in range(5):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            for o in range(2):
                                id = ((((((i*5+j)*2+k)*2+l)*2+m)*2+n)*2+o)
                                for rep in range(8):
                                    results[0].append(id)
                                    tid = id*8+rep
                                    fname = basename+"_"+str(tid).zfill(4)+".json"
                                    with open(fname,"r") as f:
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
                                    fname= basename+"_"+str(tid).zfill(4)+"_results.txt"
                                    try:
                                        with open(fname, "r") as f:
                                            d = np.loadtxt(f)
                                        results[1].append(d.shape[0])
                                        for q in range(1,5):
                                            results[q+1].append(np.mean(d[-N_avg:,q]))
                                        tt = d[-1,-1]/(d[-1,0]+1)
                                        tt /= (p["N_TRAIN"]+2264)
                                        results[11].append(tt)
                                    except:
                                        for q in range(0,5):
                                            results[q+1].append(0)
                                        results[11].append(0)
                                    fname = basename+"_"+str(tid).zfill(4)+"_best.txt"
                                    try:
                                        with open(fname, "r") as f:
                                            d= np.loadtxt(f)
                                        results[6].append(d[0])
                                        for q in range(1,5):
                                            results[q+6].append(d[q])
                                    except:
                                        for q in range(0,5):
                                                results[q+6].append(0)
    for x in results:
        print(len(x))
    results= np.asarray(results)
    print(results.shape)
    return results
