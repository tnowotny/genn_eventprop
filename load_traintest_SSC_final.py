import numpy as np
import json

def load_traintest_SSC_final(N_avg):
    # this comes from optimising lbd for all other conditions in plot_scan_results_traineval.py on scan_SSC_final
    lbd_id = [0, 12, 24, 30, 48, 50, 62, 70, 86, 96, 108, 118, 128, 136, 140, 154, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 262, 274, 280, 290, 306, 318, 328, 330, 340, 350, 366, 378, 388, 392, 406, 416, 428, 436, 446, 458, 468, 478, 480, 490, 500, 510, 520, 530, 540, 550, 566, 576, 586, 596, 608, 618, 626, 636, 644, 650, 660, 674, 684, 694, 700, 712, 726, 734, 746, 756, 766, 774, 784, 794, 804, 814, 820, 830, 846, 856, 860, 870, 886, 896, 908, 918, 926, 934, 944, 952, 960, 974, 982, 996, 1004, 1012, 1022, 1032, 1044, 1054, 1064, 1072, 1082, 1094, 1102, 1114, 1124, 1136, 1140, 1152, 1164, 1176, 1180, 1192, 1206, 1214, 1226, 1234, 1244, 1254, 1264, 1274, 1282, 1294, 1304, 1312, 1320, 1332, 1344, 1352, 1362, 1372, 1384, 1392, 1400, 1410, 1420, 1430, 1444, 1454, 1462, 1472, 1484, 1494, 1502, 1512, 1524, 1534, 1544, 1554, 1562, 1572, 1584, 1592, 1604, 1610, 1620, 1634, 1648, 1658, 1662, 1676, 1686, 1698, 1708, 1718, 1726, 1738, 1742, 1756, 1760, 1770, 1780, 1790, 1800, 1812, 1820, 1830, 1840, 1850, 1866, 1874, 1884, 1890, 1906, 1914, 1924, 1932, 1942, 1950, 1966, 1976, 1984, 1996, 2006, 2016, 2022, 2036, 2046, 2056, 2068, 2070, 2080, 2092, 2100, 2110, 2124, 2138, 2142, 2154, 2168, 2176, 2186, 2196, 2206, 2218, 2226, 2238, 2242, 2256, 2266, 2272, 2284, 2296, 2300, 2314, 2326, 2336, 2344, 2354, 2364, 2374, 2384, 2396, 2406, 2416, 2420, 2430, 2444, 2450, 2460, 2474, 2486, 2496, 2506, 2516, 2524, 2536, 2546, 2556, 2564, 2574, 2582, 2592, 2602, 2614, 2622, 2634, 2640, 2652, 2664, 2674, 2682, 2694, 2700, 2710, 2724, 2736, 2740, 2754, 2760, 2772, 2784, 2794, 2802, 2814, 2824, 2834, 2844, 2854, 2864, 2874, 2882, 2890, 2904, 2914, 2920, 2930, 2940, 2952, 2960, 2972, 2980, 2992, 3002, 3012, 3020, 3030, 3044, 3054, 3062, 3072, 3082, 3092, 3102, 3112, 3122, 3134, 3144, 3152, 3160, 3172, 3180, 3192, 3208, 3218, 3224, 3238, 3240, 3258, 3268, 3276, 3288, 3298, 3308, 3316, 3320, 3336, 3340, 3354, 3360, 3378, 3380, 3390, 3406, 3416, 3422, 3430, 3446, 3456, 3462, 3474, 3484, 3494, 3504, 3514, 3520, 3532, 3544, 3552, 3560, 3574, 3580, 3592, 3604, 3614, 3620, 3634, 3640, 3650, 3660, 3670, 3680, 3694, 3702, 3712, 3720, 3734, 3742, 3754, 3760, 3776, 3784, 3796, 3802, 3812, 3820, 3830, 3848, 3858, 3866, 3874, 3888, 3898, 3906, 3918, 3928, 3938, 3946, 3958, 3968, 3978, 3984, 3994, 4006, 4016, 4024, 4036, 4044, 4054, 4064, 4074, 4082, 4096, 4102, 4116, 4124, 4134, 4140, 4156, 4160, 4176, 4180, 4196, 4200, 4216, 4220, 4236, 4240, 4250, 4260, 4270, 4280, 4296, 4300, 4310, 4320, 4336, 4342, 4352, 4364, 4372, 4380, 4394, 4402, 4416, 4424, 4434, 4442, 4454, 4462, 4472, 4480, 4496, 4504, 4516, 4526, 4536, 4542, 4552, 4566, 4576, 4586, 4596, 4606, 4618, 4624, 4634, 4646, 4656, 4666, 4676, 4684, 4696, 4706, 4716, 4726, 4736, 4746, 4756, 4766, 4776, 4786, 4796, 4806, 4814, 4826, 4836, 4840, 4856, 4868, 4878, 4884, 4896, 4906, 4916, 4926, 4936, 4946, 4958, 4966, 4976, 4986, 4994, 5006, 5016, 5022, 5032, 5044, 5056, 5064, 5074, 5086, 5096, 5106, 5114]
    
    dt_ms= [ 1, 2 ]
    sizes1 = [ 256, 1024 ]
    sizes2 = [ 64, 128, 512 ]
    num_hidden= [ 64, 128, 256, 512, 1024 ]
    delay = [ 0, 10 ]
    shift = [ 0.0, 40.0 ]
    blend = [ [], [0.5, 0.5] ]
    train_tau = [False, True]
    hid_neuron = ["LIF", "hetLIF"]

    def load_results_internal(results, tid, fname):
        #print(f"tid: {tid}, fname: {fname}")
        with open(fname+"_"+str(tid).zfill(4)+".json","r") as f:
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
        try:
            with open(fname+"_"+str(tid).zfill(4)+"_results.txt", "r") as f:
                d = np.loadtxt(f)
                results[1].append(d.shape[0])
                for q in range(1,5):
                    results[q+1].append(np.mean(d[-N_avg:,q]))
                    tt = d[-1,-1]/(d[-1,0]+1)
                    tt /= (p["N_TRAIN"]+20382)
                results[11].append(tt)
        except:
            print(f"failed to open {fname+'_'+str(tid).zfill(4)+'_results.txt'}")
            for q in range(0,5):
                results[q+1].append(0)
            results[11].append(0)
        try:
            with open(fname+"_"+str(tid).zfill(4)+"_test_results.txt", "r") as f:
                d= np.loadtxt(f)
            results[6].append(d[0])
            for q in range(1,5):
                results[q+6].append(d[q])
        except:
            print(f"failed to open {fname+'_'+str(tid).zfill(4)+'_test_results.txt'}")
            for q in range(0,5):
                results[q+6].append(0)
        return results
    
    res_col = 12
    results = [ [] for i in range(res_col) ] # 12 types of results
    bname_scan = "scan_SSC_final/SSC"
    bname_rep = "scan_SSC_final_repeats/SSC"
    for i in range(2):
        for j in range(5):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            for o in range(2):
                                id = ((((((i*5+j)*2+k)*2+l)*2+m)*2+n)*2+o)
                                id1 = lbd_id[id]
                                fname = bname_scan
                                for rep in range(2):
                                    results[0].append(id1+rep)
                                    results= load_results_internal(results, id1+rep, fname)
                                # now do the dedicate 6 reps to make a full 8 reps
                                fname = bname_rep
                                for s in range(6):
                                    results[0].append(id*6+s)
                                    results= load_results_internal(results, id*6+s, fname)
                                    
                                    
    for x in results:
        print(len(x))
    results= np.asarray(results)
    print(results.shape)
    return results
