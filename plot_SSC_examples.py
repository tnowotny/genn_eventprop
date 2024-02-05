from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

p["DATA_SET"]= "SSC"
p["RESCALE_X"]= 1.0
p["RESCALE_T"]= 1.0
p["TRIAL_MS"]= 1000
p["DT_MS"]= 1
mn= SHD_model(p)
mn.plot_examples([],int(sys.argv[1]),10,"train")

