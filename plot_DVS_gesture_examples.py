from simulator_DVS_gesture import *
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

mn= DVSG_model(p)
mn.plot_examples(int(sys.argv[1]),10,"train")

