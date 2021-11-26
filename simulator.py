import numpy as np

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from Dataset import YinYangDataset
from models import *
import os

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
output_dir= "."
TRAIN= False
TIMESTEP_MS = 0.1
BUILD = True
TIMING = True
DATA_SEED= 123

# Experiment parameters
TRIAL_MS= 30.0
N_MAX_SPIKE= 20    # make buffers for maximally 20 spikes (10 in a 30 ms trial) - should be safe
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
V_THRESH = 1.0
V_RESET = 0.0
TAU_0= 0.5
TAU_1= 6.4
ALPHA= 3e-3

# Learning parameters
ETA= 5e-3
ADAM_BETA1= 0.9      # TODO: implement Adam optimizer
ADAM_BETA2= 0.999    # TODO: implement Adam optimizer
ADAM_EPS= 1e-8       # TODO: implement Adam optimizer
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

def update_adam(learning_rate, adam_step, optimiser_custom_updates):
    first_moment_scale = 1.0 / (1.0 - (ADAM_BETA1 ** adam_step))
    second_moment_scale = 1.0 / (1.0 - (ADAM_BETA2 ** adam_step))

    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["alpha"].view[:] = learning_rate
        o.extra_global_params["firstMomentScale"].view[:] = first_moment_scale
        o.extra_global_params["secondMomentScale"].view[:] = second_moment_scale

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
Y_train= Y_train.flatten()

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
Y_test= Y_test.flatten()

input_end_test = np.arange(chunk,NUM_INPUT*N_BATCH*chunk+1, chunk)
input_end_test = np.reshape(input_end_test, (N_BATCH, NUM_INPUT))
input_start_test = get_input_start(input_end_test)


# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------

input_params= {}
if TRAIN:
    input_init_vars= {"startSpike": input_start_train, "endSpike": input_end_train}
else:
    input_init_vars= {"startSpike": input_start_test, "endSpike": input_end_test}
    
hidden_params= {"tau_m": TAU_MEM,
                "V_thresh": V_THRESH,
                "V_reset": V_RESET,
                "N_max_spike": N_MAX_SPIKE,
                "tau_syn": TAU_SYN,
                }
hidden_init_vars= {"V": V_RESET,
                   "lambda_V": 0.0,
                   "lambda_I": 0.0,
                   "rev_t": 0.0,
                   "rp_ImV": 0,
                   "wp_ImV": 0,
                   "back_spike": 0,
                   }

output_params= {"tau_m": TAU_MEM,
                "V_thresh": V_THRESH,
                "V_reset": V_RESET,
                "N_max_spike": N_MAX_SPIKE,
                "tau_syn": TAU_SYN,
                "trial_t": TRIAL_MS,
                "tau0": TAU_0,
                "tau1": TAU_1,
                "alpha": ALPHA,
                "N_batch": N_BATCH,
                }

output_init_vars= {"V": V_RESET,
                   "lambda_V": 0.0,
                   "lambda_I": 0.0,
                   "rev_t": 0.0,
                   "rp_ImV": 0,
                   "wp_ImV": 0,
                   "back_spike": 0,
                   "first_spike_t": -1e5,
                   "new_first_spike_t": -1e5,
                   "expsum": 1.0,
                   }

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------

INPUT_HIDDEN_MEAN= 1.5
INPUT_HIDDEN_STD= 0.78

in_to_hid_init_vars= {"dw": 0}
in_to_hid_init_vars["w"]= genn_model.init_var("Normal", {"mean": INPUT_HIDDEN_MEAN, "sd": INPUT_HIDDEN_STD})

HIDDEN_OUTPUT_MEAN= 0.93
HIDDEN_OUTPUT_STD= 0.1
hid_to_out_init_vars= {"dw": 0}
hid_to_out_init_vars["w"]= genn_model.init_var("Normal", {"mean": HIDDEN_OUTPUT_MEAN, "sd": HIDDEN_OUTPUT_STD})

# ----------------------------------------------------------------------------
# Optimiser initialisation
# ----------------------------------------------------------------------------

adam_params = {"beta1": ADAM_BETA1, "beta2": ADAM_BETA2, "epsilon": 1E-8, "tau_syn": TAU_SYN, "N_batch": N_BATCH}
adam_init_vars = {"m": 0.0, "v": 0.0}

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------

model = genn_model.GeNNModel("float", "eventprop_yingyang", generateLineInfo=True)
model.dT = TIMESTEP_MS
model.timing_enabled = TIMING
model.batch_size = N_BATCH

# Add neuron populations
input = model.add_neuron_population("Input", NUM_INPUT, "SpikeSourceArray", 
                                    {}, input_init_vars)
if TRAIN:
    input.set_extra_global_param("spikeTimes", X_train)
else:
    input.set_extra_global_param("spikeTimes", X_test)
                                    
hidden= model.add_neuron_population("hidden", NUM_HIDDEN, EVP_LIF, hidden_params, hidden_init_vars) 
hidden.set_extra_global_param("t_k",-1e5*np.ones(NUM_HIDDEN*N_MAX_SPIKE,dtype=np.float32))
hidden.set_extra_global_param("ImV",np.zeros(NUM_HIDDEN*N_MAX_SPIKE,dtype=np.float32))

output= model.add_neuron_population("output", NUM_OUTPUT, EVP_LIF_output, output_params, output_init_vars)
output.set_extra_global_param("t_k",-1e5*np.ones(NUM_OUTPUT*N_MAX_SPIKE,dtype=np.float32))
output.set_extra_global_param("ImV",np.zeros(NUM_OUTPUT*N_MAX_SPIKE,dtype=np.float32))
output.set_extra_global_param("label", 0)

hidden_var_refs= {"rp_ImV": genn_model.create_var_ref(hidden, "rp_ImV"),
                  "wp_ImV": genn_model.create_var_ref(hidden, "wp_ImV"),
                  "V": genn_model.create_var_ref(hidden, "V"),
                  "lambda_V": genn_model.create_var_ref(hidden, "lambda_V"),
                  "lambda_I": genn_model.create_var_ref(hidden, "lambda_I"),
                  "rev_t": genn_model.create_var_ref(hidden, "rev_t"),
                  "back_spike": genn_model.create_var_ref(hidden, "back_spike")
                  }
hidden_reset=  model.add_custom_update("hidden_reset","neuronReset", EVP_neuron_reset, {"V_reset": V_RESET, "N_max_spike": N_MAX_SPIKE}, {}, hidden_var_refs)

output_reset_params= {"V_reset": V_RESET,
                      "N_max_spike": N_MAX_SPIKE,
                      "tau0": TAU_0,
                      "tau1": TAU_1
                      }
output_var_refs= {"rp_ImV": genn_model.create_var_ref(output, "rp_ImV"),
                  "wp_ImV": genn_model.create_var_ref(output, "wp_ImV"),
                  "V": genn_model.create_var_ref(output, "V"),
                  "lambda_V": genn_model.create_var_ref(output, "lambda_V"),
                  "lambda_I": genn_model.create_var_ref(output, "lambda_I"),
                  "rev_t": genn_model.create_var_ref(output, "rev_t"),
                  "back_spike": genn_model.create_var_ref(output, "back_spike"),
                  "first_spike_t": genn_model.create_var_ref(output, "first_spike_t"),
                  "new_first_spike_t": genn_model.create_var_ref(output, "new_first_spike_t"),
                  "expsum": genn_model.create_var_ref(output, "expsum")
                  }
output_reset=  model.add_custom_update("output_reset","neuronResetOutput", EVP_neuron_reset_output, output_reset_params, {}, output_var_refs)


# synapse populations
in_to_hid= model.add_synapse_population("in_to_hid", "DENSE_INDIVIDUALG", NO_DELAY, input, hidden, EVP_input_synapse,
                                             {}, in_to_hid_init_vars, {}, {}, "ExpCurr", {"tau": TAU_SYN}, {}
                                            )
hid_to_out= model.add_synapse_population("hid_to_out", "DENSE_INDIVIDUALG", NO_DELAY, hidden, output, EVP_synapse,
                                              {}, hid_to_out_init_vars, {}, {}, "ExpCurr", {"tau": TAU_SYN}, {}
                                            )

var_refs = {"dw": genn_model.create_wu_var_ref(in_to_hid, "dw")}
in_to_hid_reduce= model.add_custom_update("in_to_hid_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
var_refs = {"gradient": genn_model.create_wu_var_ref(in_to_hid_reduce, "reduced_dw"),
            "variable": genn_model.create_wu_var_ref(in_to_hid, "w")}
in_to_hid_learn= model.add_custom_update("in_to_hid_learn","EVPLearn", adam_optimizer_model, adam_params, adam_init_vars, var_refs)

var_refs = {"dw": genn_model.create_wu_var_ref(hid_to_out, "dw")}
hid_to_out_reduce= model.add_custom_update("hid_to_out_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
var_refs = {"gradient": genn_model.create_wu_var_ref(hid_to_out_reduce, "reduced_dw"),
            "variable": genn_model.create_wu_var_ref(hid_to_out, "w")}
hid_to_out_learn= model.add_custom_update("hid_to_out_learn","EVPLearn", adam_optimizer_model, adam_params, adam_init_vars, var_refs)

optimisers= [in_to_hid_learn, hid_to_out_learn]

model.build()
model.load()

# ----------------------------------------------------------------------------
# Simulation loop
# ----------------------------------------------------------------------------


if TRAIN:
    in_to_hid.pull_var_from_device("w")
    hid_to_out.pull_var_from_device("w")
    np.save(os.path.join(output_dir, "w_input_hidden_0.npy"), in_to_hid.vars["w"].view.copy())
    np.save(os.path.join(output_dir, "w_hidden_output_0.npy"), hid_to_out.vars["w"].view.copy())
else:
    in_to_hid.vars["w"].view[:]= np.load(os.path.join(output_dir, "w_input_hidden_1.npy"))
    hid_to_out.vars["w"].view[:]= np.load(os.path.join(output_dir, "w_hidden_output_1.npy"))
    in_to_hid.push_var_to_device("w")
    hid_to_out.push_var_to_device("w")
    
if TRAIN:
    N_trial= (N_TRAIN * N_CLASS) // N_BATCH
else:
    N_trial= (N_TEST * N_CLASS) // N_BATCH

good= 0.0    
fst= output.vars["first_spike_t"].view
adam_step= 1
learning_rate= ETA
cnt= np.ones(N_BATCH)

for trial in range(N_trial):
    if TRAIN:
        output.set_extra_global_param("label", int(Y_train[trial]))
    else:
        output.set_extra_global_param("label", int(Y_test[trial]))
    trial_end= (trial+1)*TRIAL_MS
    while (model.t <  trial_end-1e-3*TIMESTEP_MS):
        model.step_time()

    if TRAIN:
        update_adam(learning_rate, adam_step, optimisers)
        adam_step += 1
        model.custom_update("EVPReduce")
        model.custom_update("EVPLearn")
    else:
        output.pull_var_from_device("first_spike_t");
        print(fst.shape)
        pred= np.argmin(fst[:,:],axis=1)
        print(pred.shape)
        good += np.sum(cnt[pred == Y_test[trial*N_BATCH:(trial+1)*N_BATCH]])
        print(good) 
    model.custom_update("neuronReset")
    model.custom_update("neuronResetOutput")

print("Correct: {}".format(good/(N_trial*N_BATCH)))
    
in_to_hid.pull_var_from_device("w")
hid_to_out.pull_var_from_device("w")
np.save(os.path.join(output_dir, "w_input_hidden_1.npy"), in_to_hid.vars["w"].view.copy())
np.save(os.path.join(output_dir, "w_hidden_output_1.npy"), hid_to_out.vars["w"].view.copy())
