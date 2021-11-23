import numpy as np

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from Dataset import YinYangDataset

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
TIMESTEP_MS = 0.1
BUILD = True
TIMING = True
DATA_SEED= 123

# Experiment parameters
TRIAL_MS= 30.0
DATA_SEED= 123
N_BATCH= 32
N_TRAIN= N_BATCH*1000
N_EPOCH= 10
N_TEST= N_BATCH*100
N_CLASS= 3

# Network structure
NUM_INPUT = 5
NUM_OUTPUT = N_CLASS
NUM_HIDDEN = 200

# Model parameters
TAU_SYN = 5.0
TAU_MEM = 20.0

# Learning parameters
ETA= 5e-3
BETA1= 0.9
BETA2= 0.999
EPS= 1e-8
# applied every epoch
ETA_DECAY= 0.95

# Convert parameters to timesteps
TRIAL_TIMESTEPS = int(TRIAL_MS / TIMESTEP_MS)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def get_input_start(input_end):
    input_start = np.empty_like(input_end)
    if input_end.ndim == 1:
        input_start[0] = 0
        input_start[1:] = input_end[:-1]
    else:
        input_start[0,0] = 0
        input_start[1:,0] = input_end[:-1,-1]
        input_start[:,1:] = input_end[:,:-1]

    return input_start

# ----------------------------------------------------------------------------
# Input and output preparation
# ----------------------------------------------------------------------------

np.random.seed(DATA_SEED)
X_train, Y_train = YinYangDataset(size=N_TRAIN * N_CLASS, 
                                  flipped_coords=True, seed=DATA_SEED)[:]
X_train= X_train.T
z= np.zeros(N_TRAIN * N_CLASS)
X_train= np.vstack([z, X_train])
chunk= (N_TRAIN * N_CLASS) // N_BATCH
X_train= np.vstack([ X_train[:,i*chunk:(i+1)*chunk] for i in range(N_BATCH)])
offset= np.reshape(np.arange(0,chunk * TRIAL_MS, TRIAL_MS),(1,chunk))
offset= np.repeat(offset,NUM_INPUT*N_BATCH,axis=0)
X_train= X_train*TRIAL_MS+offset
X_train= X_train.flatten()

Y_train= np.vstack([ Y_train[i*chunk:(i+1)*chunk] for i in range(N_BATCH)])

input_end_train = np.arange(chunk,NUM_INPUT*N_BATCH*chunk+1, chunk)
input_end_train = np.reshape(input_end_train, (N_BATCH, NUM_INPUT))
input_start_train = get_input_start(input_end_train)

X_test, Y_test = YinYangDataset(size=N_TEST * N_CLASS, 
                                flipped_coords=True, seed=None)[:]
X_test= X_test.T
z= np.zeros(N_TEST * N_CLASS)
X_test= np.vstack([z, X_test])
chunk= (N_TEST * N_CLASS) // N_BATCH
X_test= np.vstack([ X_test[:,i*chunk:(i+1)*chunk] for i in range(N_BATCH)])
offset= np.reshape(np.arange(0,chunk * TRIAL_MS, TRIAL_MS),(1,chunk))
offset= np.repeat(offset,NUM_INPUT*N_BATCH,axis=0)
X_test= X_test*TRIAL_MS+offset
X_test= X_test.flatten()

Y_test= np.vstack([ Y_test[i*chunk:(i+1)*chunk] for i in range(N_BATCH)])

input_end_test = np.arange(chunk,NUM_INPUT*N_BATCH*chunk+1, chunk)
input_end_test = np.reshape(input_end_test, (N_BATCH, NUM_INPUT))
input_start_test = get_input_start(input_end_test)


# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------

input_params= {}
input_init_vars= {"startSpike": input_start_train, "endSpike": input_end_train}

hidden_params= {}
hidden_init_vars= {}

output_params= {}
output_init_vars= {}

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------

INPUT_HIDDEN_MEAN= 1.5
INPUT_HIDDEN_STD= 0.78
input_hidden_weight_dist_params = {"mean": INPUT_HIDDEN_MEAN, "sd": INPUT_HIDDEN_STD,
                                   "min": INPUT_HIDDEN_MEAN-3*INPUT_HIDDEN_STD,
                                   "max": INPUT_HIDDEN_MEAN+3*INPUT_HIDDEN_STD}

HIDDEN_OUTPUT_MEAN= 0.93
HIDDEN_OUTPUT_STD= 0.1
hidden_output_weight_dist_params = {"mean": HIDDEN_OUTPUT_MEAN, "sd": HIDDEN_OUTPUT_STD,
                                   "min": HIDDEN_OUTPUT_MEAN-3*HIDDEN_OUTPUT_STD,
                                   "max": HIDDEN_OUTPUT_MEAN+3*HIDDEN_OUTPUT_STD}


# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------

model = genn_model.GeNNModel("float", "eventprop_yingyang", generateLineInfo=True)
model.dT = TIMESTEP_MS
model.timing_enabled = TIMING



# Add neuron populations
input = model.add_neuron_population("Input", NUM_INPUT, "SpikeSourceArray", 
                                    {}, input_init_vars)
input.set_extra_global_parameter("spikeTimes", X_train)
                                    
hidden= genn_model.add_neuron_population("hidden", n_hidden, EVP_LIF, hidden_params, hidden_init_vars) 
output= genn_model.add_neuron_population("output", n_output, EVP_LIF_output, output_params, output_init_vars)

# synapse populations
in_to_hid= genn_model.add_synapse_population("in_to_hid", "DENSE_INDIVIDUALG", NO_DELAY, input, hidden, EVP_synapse,
                                             {}, {"w": 0, "dw": 0.0}, "ExpCurr", {"tau_syn": tau_syn}, {}
                                            )
