from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

p["DATASET"]= "SHD"
p["RESCALE_X"]= 1.0
p["RESCALE_T"]= 0.1
p["TRIAL_MS"]= 100
mn= SHD_model(p)
mn.plot_examples([0,1,2,3,6,7,8,9,10,11],int(sys.argv[1]),10,"train")

