import os
import json
import numpy as np

bname1= "scan_SSC_JUWELS_18/JSSC18_scan_"
bname2= "scan_SSC_JUWELS_18b/JSSC18b_scan_"

seeds_add = list(np.asarray(range(8))*11+22)
print(seeds_add)
num_hidden= [ 64, 128, 256, 512, 1024 ]
sizes1 = [ 256, 1024 ]
sizes2 = [ 64, 128, 512 ]

# this comes from optimising lbd for all other conditions in plot_scan_results_traineval.py on scan_SSC_JUWELS_18
lbd_id1 = [4, 10, 20, 34, 44, 54, 60, 72, 86, 94, 106, 116, 126, 134, 144, 154, 164, 174, 180, 190, 206, 216, 220, 230, 246, 256, 268, 278, 286, 294, 304, 312, 322, 334, 344, 352, 360, 372, 384, 392, 402, 412, 424, 432, 440, 450, 460, 470, 484, 494, 502, 512, 524, 534, 542, 552, 564, 574, 584, 594, 602, 612, 624, 632, 642, 656, 666, 672, 684, 696, 700, 714, 726, 736, 744, 754, 764, 774, 784, 796, 806, 816, 820, 830, 844, 850, 860, 874, 886, 896, 906, 916, 924, 936, 946, 956, 962, 970, 984, 994, 1000, 1010, 1020, 1032, 1040, 1052, 1060, 1072, 1082, 1092, 1100, 1110, 1124, 1134, 1142, 1152, 1162, 1172, 1182, 1192, 1202, 1214, 1224, 1232, 1240, 1252, 1260, 1272, 1288, 1298, 1304, 1318, 1320, 1338, 1348, 1356, 1368, 1378, 1388, 1396, 1400, 1416, 1420, 1434, 1440, 1458, 1460, 1470, 1486, 1496, 1502, 1510, 1526, 1536, 1542, 1554, 1564, 1574, 1584, 1594, 1600, 1612, 1624, 1632, 1640, 1654, 1660, 1672, 1684, 1694, 1700, 1714, 1720, 1730, 1740, 1750, 1760, 1774, 1782, 1792, 1800, 1814, 1822, 1834, 1840, 1856, 1864, 1876, 1882, 1892, 1900, 1910, 1928, 1938, 1946, 1954, 1968, 1978, 1986, 1998, 2008, 2018, 2026, 2038, 2048, 2058, 2064, 2074, 2086, 2096, 2104, 2116, 2124, 2134, 2144, 2154, 2162, 2176, 2182, 2196, 2204, 2214, 2220, 2236, 2240, 2256, 2260, 2276, 2280, 2296, 2300, 2316, 2320, 2330, 2340, 2350, 2360, 2376, 2380, 2390, 2400, 2416, 2422, 2432, 2444, 2452, 2460, 2474, 2482, 2496, 2504, 2514, 2522, 2534, 2542, 2556, 2560, 2576, 2584, 2596, 2606, 2616, 2622, 2632, 2646, 2656, 2666, 2676, 2686, 2698, 2704, 2714, 2726, 2736, 2746, 2756, 2764, 2776, 2786, 2796, 2806, 2816, 2826, 2836, 2846, 2856, 2866, 2876, 2886, 2894, 2906, 2916, 2920, 2936, 2948, 2958, 2964, 2976, 2986, 2996, 3006, 3016, 3026, 3038, 3046, 3056, 3066, 3074, 3086, 3096, 3102, 3112, 3124, 3136, 3144, 3154, 3166, 3176, 3186, 3194]

# this comes from optimising lbd for all other conditions in plot_scan_results_traineval.py on scan_SSC_JUWELS_18b
lbd_id2 = [0, 10, 26, 38, 48, 58, 66, 76, 84, 94, 104, 114, 124, 130, 146, 154, 166, 178, 186, 198, 204, 214, 224, 234]

k=1
l=1
m=1
for i in range(2):
    for j in range(5):
        for n in range(2):
            for o in range(2):
                if num_hidden[j] in sizes1:
                    the_j = np.where(np.asarray(sizes1) == num_hidden[j])[0][0]
                    id1 = lbd_id1[((((((i*2+the_j)*2+k)*2+l)*2+m)*2+n)*2+o)]
                    fname = bname1+str(id1)+".json"
                else:
                    the_j= np.where(np.asarray(sizes2) == num_hidden[j])[0][0]
                    id1 = lbd_id2[(((i*3+the_j)*2+n)*2+o)]
                    fname = bname2+str(id1)+".json"
                print(fname)
                with open(fname,"r") as f:
                    p0= json.load(f)
                for s in range(6):
                    id0 = (((i*5+j)*2+n)*2+o)*6+s
                    print(id0)
                    p = p0
                    p["ORIG_NAME"] = fname
                    p["TRAIN_DATA_SEED"] = int(p["TRAIN_DATA_SEED"]+seeds_add[s])
                    p["TEST_DATA_SEED"] = int(p["TEST_DATA_SEED"]+seeds_add[s])
                    p["MODEL_SEED"] = int(p["MODEL_SEED"]+seeds_add[s])
                    p["SPK_REC_STEPS"] = int(p["TRIAL_MS"]/p["DT_MS"])
                    p["OUT_DIR"] = "scan_SSC_JUWELS_19/"
                    p["NAME"] = "JSSC19_scan_"+str(id0)
                    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
                        json.dump(p, f)
