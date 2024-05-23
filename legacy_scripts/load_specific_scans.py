import numpy as np
import json

def load_train_test_48(N_avg):

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
    basename = "scan_JUWELS_48/J48_scan"
    basename_2 = "scan_JUWELS_48c/J48c_scan"
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
                                if k == 1 and l == 1 and m == 1:
                                    bname= basename
                                    id = ((i*5+j)*2+n)*2+o
                                else:
                                    bname = basename_2
                                for rep in range(8):
                                    tid = id*8+rep
                                    fname = bname+"_"+str(tid)+".json"
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
                                    fname= bname+"_"+str(tid)+"_results.txt"
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
                                    fname = bname+"_"+str(tid)+"_best.txt"
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



def load_SSC_test_19(N_avg):
    # this comes from optimising lbd for all other conditions in plot_scan_results_traineval.py on scan_SSC_JUWELS_18
    lbd_id1 = [4, 10, 20, 34, 44, 54, 60, 72, 86, 94, 106, 116, 126, 134, 144, 154, 164, 174, 180, 190, 206, 216, 220, 230, 246, 256, 268, 278, 286, 294, 304, 312, 322, 334, 344, 352, 360, 372, 384, 392, 402, 412, 424, 432, 440, 450, 460, 470, 484, 494, 502, 512, 524, 534, 542, 552, 564, 574, 584, 594, 602, 612, 624, 632, 642, 656, 666, 672, 684, 696, 700, 714, 726, 736, 744, 754, 764, 774, 784, 796, 806, 816, 820, 830, 844, 850, 860, 874, 886, 896, 906, 916, 924, 936, 946, 956, 962, 970, 984, 994, 1000, 1010, 1020, 1032, 1040, 1052, 1060, 1072, 1082, 1092, 1100, 1110, 1124, 1134, 1142, 1152, 1162, 1172, 1182, 1192, 1202, 1214, 1224, 1232, 1240, 1252, 1260, 1272, 1288, 1298, 1304, 1318, 1320, 1338, 1348, 1356, 1368, 1378, 1388, 1396, 1400, 1416, 1420, 1434, 1440, 1458, 1460, 1470, 1486, 1496, 1502, 1510, 1526, 1536, 1542, 1554, 1564, 1574, 1584, 1594, 1600, 1612, 1624, 1632, 1640, 1654, 1660, 1672, 1684, 1694, 1700, 1714, 1720, 1730, 1740, 1750, 1760, 1774, 1782, 1792, 1800, 1814, 1822, 1834, 1840, 1856, 1864, 1876, 1882, 1892, 1900, 1910, 1928, 1938, 1946, 1954, 1968, 1978, 1986, 1998, 2008, 2018, 2026, 2038, 2048, 2058, 2064, 2074, 2086, 2096, 2104, 2116, 2124, 2134, 2144, 2154, 2162, 2176, 2182, 2196, 2204, 2214, 2220, 2236, 2240, 2256, 2260, 2276, 2280, 2296, 2300, 2316, 2320, 2330, 2340, 2350, 2360, 2376, 2380, 2390, 2400, 2416, 2422, 2432, 2444, 2452, 2460, 2474, 2482, 2496, 2504, 2514, 2522, 2534, 2542, 2556, 2560, 2576, 2584, 2596, 2606, 2616, 2622, 2632, 2646, 2656, 2666, 2676, 2686, 2698, 2704, 2714, 2726, 2736, 2746, 2756, 2764, 2776, 2786, 2796, 2806, 2816, 2826, 2836, 2846, 2856, 2866, 2876, 2886, 2894, 2906, 2916, 2920, 2936, 2948, 2958, 2964, 2976, 2986, 2996, 3006, 3016, 3026, 3038, 3046, 3056, 3066, 3074, 3086, 3096, 3102, 3112, 3124, 3136, 3144, 3154, 3166, 3176, 3186, 3194]

    # this comes from optimising lbd for all other conditions in plot_scan_results_traineval.py on scan_SSC_JUWELS_18b
    lbd_id2 = [0, 10, 26, 38, 48, 58, 66, 76, 84, 94, 104, 114, 124, 130, 146, 154, 166, 178, 186, 198, 204, 214, 224, 234]

    # this comes from optimising lbd for all other conditions in plot_scan_results_xval.py on scan_SSC_JUWELS_18c
    lbd_id3 = [0, 12, 24, 30, 48, 50, 62, 70, 86, 96, 108, 118, 128, 136, 140, 154, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 262, 274, 280, 290, 300, 310, 328, 330, 340, 350, 366, 378, 388, 392, 406, 416, 428, 436, 446, 458, 468, 478, 480, 490, 500, 510, 520, 530, 540, 550, 566, 576, 586, 596, 600, 610, 620, 630, 640, 654, 662, 676, 684, 692, 702, 712, 724, 734, 744, 752, 762, 774, 782, 794, 804, 816, 820, 832, 844, 856, 860, 872, 886, 894, 906, 914, 920, 930, 940, 950, 964, 970, 980, 994, 1008, 1018, 1022, 1036, 1046, 1058, 1068, 1078, 1086, 1098, 1102, 1116, 1120, 1130, 1140, 1150, 1160, 1172, 1180, 1190, 1200, 1210, 1226, 1234, 1240, 1250, 1260, 1270, 1284, 1292, 1302, 1310, 1326, 1336, 1344, 1356, 1366, 1376, 1382, 1396, 1406, 1416, 1428, 1430, 1440, 1452, 1460, 1470, 1484, 1498, 1502, 1514, 1528, 1536, 1546, 1556, 1560, 1570, 1580, 1590, 1604, 1614, 1622, 1632, 1642, 1654, 1662, 1674, 1680, 1692, 1704, 1714, 1722, 1734, 1740, 1750, 1764, 1776, 1780, 1794, 1800, 1812, 1824, 1834, 1842, 1854, 1864, 1874, 1880, 1890, 1900, 1910]
 
    
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
        with open(fname+"_"+str(tid)+".json","r") as f:
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
            with open(fname+"_"+str(tid)+"_results.txt", "r") as f:
                d = np.loadtxt(f)
                results[1].append(d.shape[0])
                for q in range(1,5):
                    results[q+1].append(np.mean(d[-N_avg:,q]))
                    tt = d[-1,-1]/(d[-1,0]+1)
                    tt /= (p["N_TRAIN"]+20382)
                results[11].append(tt)
        except:
            print(f"failed to open {fname+'_'+str(tid)+'_results.txt'}")
            for q in range(0,5):
                results[q+1].append(0)
            results[11].append(0)
        try:
            with open(fname+"_"+str(tid)+"_test_results.txt", "r") as f:
                d= np.loadtxt(f)
            results[6].append(d[0])
            for q in range(1,5):
                results[q+6].append(d[q])
        except:
            print(f"failed to open {fname+'_'+str(tid)+'_test_results.txt'}")
            for q in range(0,5):
                results[q+6].append(0)
        return results
    
    res_col = 12
    results = [ [] for i in range(res_col) ] # 12 types of results
    #split= [ 2, 5, 2, 2, 2, 2, 2, 8 ]
    #s= np.prod(split)
    bname_scan = "scan_SSC_JUWELS_18/JSSC18_scan"
    bname_scan_2 = "scan_SSC_JUWELS_18b/JSSC18b_scan"
    bname_scan_3 = "scan_SSC_JUWELS_18c/JSSC18c_scan"
    bname = "scan_SSC_JUWELS_19/JSSC19_scan"
    bname_2 = "scan_SSC_JUWELS_19c/JSSC19c_scan"
    for i in range(2):
        for j in range(5):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            for o in range(2):
                                id = ((((((i*5+j)*2+k)*2+l)*2+m)*2+n)*2+o)
                                if (k == 1 and l == 1 and m == 1):
                                    if num_hidden[j] in sizes1:
                                        the_j = np.where(np.asarray(sizes1) == num_hidden[j])[0][0]
                                        id1 = lbd_id1[((((((i*2+the_j)*2+k)*2+l)*2+m)*2+n)*2+o)]
                                        fname = bname_scan
                                    else:
                                        the_j= np.where(np.asarray(sizes2) == num_hidden[j])[0][0]
                                        id1 = lbd_id2[(((i*3+the_j)*2+n)*2+o)]
                                        fname = bname_scan_2
                                else:
                                    if num_hidden[j] in sizes1:
                                        the_j = np.where(np.asarray(sizes1) == num_hidden[j])[0][0]
                                        id1 = lbd_id1[((((((i*2+the_j)*2+k)*2+l)*2+m)*2+n)*2+o)]
                                        fname = bname_scan
                                    else:
                                        the_j= np.where(np.asarray(sizes2) == num_hidden[j])[0][0]
                                        id1 = lbd_id3[((((((i*3+the_j)*2+k)*2+l)*2+m)*2+n)*2+o)]
                                        fname = bname_scan_3
                                for rep in range(2):
                                    results[0].append(id1+rep)
                                    results= load_results_internal(results, id1+rep, fname)
                                # now do the dedicate 6 reps to make a full 8 reps
                                if (k == 1 and l == 1 and m == 1):
                                    id1 = ((i*5+j)*2+n)*2+o
                                    fname = bname
                                else:
                                    id1 = (((((i*5+j)*2+k)*2+l)*2+m)*2+n)*2+o
                                    fname = bname_2
                                for s in range(6):
                                    results[0].append(id1*6+s)
                                    results= load_results_internal(results, id1*6+s, fname)
                                    
                                    
    for x in results:
        print(len(x))
    results= np.asarray(results)
    print(results.shape)
    return results
