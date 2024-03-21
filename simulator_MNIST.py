import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from utils import random_shift, random_dilate, ID_jitter
import mnist
from models import *
import os
import urllib.request
import gzip, shutil
from tensorflow.keras.utils import get_file
import tables
import copy
from time import perf_counter


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

p= {}
p["NAME"]= "test"
p["DEBUG_HIDDEN_N"]= False
p["OUT_DIR"]= "."
p["DT_MS"] = 1.0
p["BUILD"] = True
p["TIMING"] = True

# Experiment parameters
p["TRAIN_DATA_SEED"]= 123
p["TEST_DATA_SEED"]= 456
p["MODEL_SEED"]= None
p["TRIAL_MS"]= 20.0
p["N_MAX_SPIKE"]= 50    # make buffers for maximally 50 spikes (25 in a 20 ms trial) - overkill at dt= 1ms
p["N_BATCH"]= 32
p["N_TRAIN"]= 55000
p["N_VALIDATE"]= 5000
p["N_EPOCH"]= 20
p["SHUFFLE"]= True
p["N_TEST"]= 10000

# Network structure
p["NUM_HIDDEN"] = 128
p["RECURRENT"] = False

# Model parameters
p["TAU_SYN"] = 5.0
p["TAU_MEM"] = 20.0
p["V_THRESH"] = 1.0
p["V_RESET"] = 0.0
p["INPUT_HIDDEN_MEAN"]= 0.078
p["INPUT_HIDDEN_STD"]= 0.045
p["HIDDEN_OUTPUT_MEAN"]= 0.2
p["HIDDEN_OUTPUT_STD"]= 0.37
p["HIDDEN_HIDDEN_MEAN"]= 0.2   # only used when recurrent
p["HIDDEN_HIDDEN_STD"]= 0.37   # only used when recurrent
p["PDROP_INPUT"] = 0.2
p["PDROP_HIDDEN"] = 0.0
p["REG_TYPE"]= "none"
p["LBD_UPPER"]= 5e-6
p["LBD_LOWER"]= 0.001
p["NU_UPPER"]= 2.0
p["RHO_UPPER"]= 5000.0
p["GLB_UPPER"]= 1e-5

# Learning parameters
p["ETA"]= 0.01
p["ADAM_BETA1"]= 0.9      
p["ADAM_BETA2"]= 0.999    
p["ADAM_EPS"]= 1e-8       

# recording
p["W_OUTPUT_EPOCH_TRIAL"] = []
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["REC_SPIKES_EPOCH_TRIAL"] = []
p["REC_SPIKES"] = []
p["REC_NEURONS_EPOCH_TRIAL"] = []
p["REC_NEURONS"] = []
p["REC_SYNAPSES_EPOCH_TRIAL"] = []
p["REC_SYNAPSES"] = []
p["WRITE_TO_DISK"]= True
p["LOAD_LAST"]= False

# possible loss types: "first_spike", "first_spike_exp", "max", "sum", "avg_xentropy"
p["LOSS_TYPE"]= "avg_xentropy"
p["EVALUATION"]= "random"
p["CUDA_VISIBLE_DEVICES"]= True
p["AVG_SNSUM"]= False
p["REDUCED_CLASSES"]= None

# "first_spike" loss function variables
p["TAU_0"]= 0.5
p["TAU_1"]= 6.4
p["ALPHA"]= 3e-3

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def update_adam(learning_rate, adam_step, optimiser_custom_updates):
    first_moment_scale = 1.0 / (1.0 - (p["ADAM_BETA1"] ** adam_step))
    second_moment_scale = 1.0 / (1.0 - (p["ADAM_BETA2"] ** adam_step))

    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["alpha"].view[:] = learning_rate
        o.extra_global_params["firstMomentScale"].view[:] = first_moment_scale
        o.extra_global_params["secondMomentScale"].view[:] = second_moment_scale

class mnist_model:

    def __init__(self, p):
        if p["TRAIN_DATA_SEED"] is not None:
            self.datarng= np.random.default_rng(p["TRAIN_DATA_SEED"])
        else:
            self.datarng= np.random.default_rng()

        if p["TEST_DATA_SEED"] is not None:
            self.tdatarng= np.random.default_rng(p["TEST_DATA_SEED"])
        else:
            self.tdatarng= np.random.default_rng()        

        self.load_data_MNIST(p)

        if p["REDUCED_CLASSES"] is not None and len(p["REDUCED_CLASSES"]) > 0:
            self.X_train_orig, self.Y_train_orig, self.Z_train_orig= self.reduce_classes(self.X_train_orig, self.Y_train_orig, self.Z_train_orig, p["REDUCED_CLASSES"])
            self.X_test_orig, self.Y_test_orig, self.Z_test_orig= self.reduce_classes(self.X_test_orig, self.Y_test_orig, self.Z_test_orig, p["REDUCED_CLASSES"])

    def loss_func_first_spike_exp(self, nfst, Y, trial, N_class, N_batch):
        t= nfst-trial*p["TRIAL_MS"]
        expsum= self.output.vars["expsum"].view[:N_batch,0]
        exp_st= self.output.vars["exp_st"].view[:N_batch,:self.N_class]
        pred= np.argmin(t,axis=-1)
        exp_st= np.array([ exp_st[i,pred[i]] for i in range(pred.shape[0])])
        selected= np.array([ t[i,pred[i]] for i in range(pred.shape[0])])
        selected[selected > p["TRIAL_MS"]]= p["TRIAL_MS"]
        loss= -np.sum(np.log(exp_st/expsum)-p["ALPHA"]*(np.exp(selected/p["TAU_1"])-1))
        loss/= N_batch
        return loss

    def loss_func_first_spike(self, nfst, Y, trial, N_class, N_batch):
        t= nfst-trial*p["TRIAL_MS"]
        expsum= self.output.vars["expsum"].view[:N_batch,0]
        exp_st= self.output.vars["exp_st"].view[:N_batch,:self.N_class]
        pred= np.argmin(t,axis=-1)
        exp_st= np.array([ exp_st[i,pred[i]] for i in range(pred.shape[0])])
        selected= np.array([ t[i,pred[i]] for i in range(pred.shape[0])])
        selected[selected > p["TRIAL_MS"]]= p["TRIAL_MS"]        
        loss= -np.sum(np.log(exp_st/expsum)-p["ALPHA"]/(1.01*p["TRIAL_MS"]-selected))
        loss/= N_batch
        return loss

    def loss_func_max(self, Y, N_batch):
        expsum= self.output.vars["expsum"].view[:N_batch,:self.N_class]
        exp_V= self.output.vars["exp_V"].view[:N_batch,:self.N_class]
        exp_V_correct= np.array([ exp_V[i,y] for i, y in enumerate(Y) ])
        if (np.sum(exp_V_correct == 0) > 0):
            print("exp_V flushed to 0 exception!")
            print(exp_V_correct)
            print(exp_V[np.where(exp_V_correct == 0),:])
            exp_V_correct[exp_V_correct == 0]+= 2e-45 # make sure all exp_V are > 0
        loss= -np.sum(np.log(exp_V_correct)-np.log(expsum[:,0]))/N_batch
        return loss

    def loss_func_sum(self, Y, N_batch):
        SoftmaxVal= self.output.vars["SoftmaxVal"].view[:N_batch,:self.N_class]
        SoftmaxVal_correct= np.array([ SoftmaxVal[i,y] for i, y in enumerate(Y) ])
        if (np.sum(SoftmaxVal_correct == 0) > 0):
            print("exp_V flushed to 0 exception!")
            print(SoftmaxVal_correct)
            print(SoftmaxVal[np.where(SoftmaxVal_correct == 0),:])
            SoftmaxVal_correct[SoftmaxVal_correct == 0]+= 2e-45 # make sure all exp_V are > 0
        loss= -np.sum(np.log(SoftmaxVal_correct))/N_batch
        return loss

    def loss_func_avg_xentropy(self, N_batch):
        loss= self.output.vars["loss"].view[:N_batch,:self.N_class]
        loss= np.mean(np.sum(loss,axis=1)) # use mean to achieve 1/N_batch here
        return loss
    
    def load_data_MNIST(self, p, shuffle= True):
        X = mnist.train_images()
        Y = mnist.train_labels()
        self.N_class= 10
        self.num_input= 28*28
        self.num_output= self.N_class 
        self.data_max_length= len(Y)+2*p["N_BATCH"]
        idx= np.arange(len(Y))
        if (shuffle):
            self.datarng.shuffle(idx)
            X= X[idx]
        self.X_val_orig= X[len(Y)-p["N_VALIDATE"]:,:,:]
        self.X_train_orig= X[:p["N_TRAIN"],:,:]
        Y= Y[idx]
        self.Y_val_orig= Y[len(Y)-p["N_VALIDATE"]:]
        self.Y_train_orig= Y[:p["N_TRAIN"]]
        # also load some testing data
        X = mnist.test_images()
        Y = mnist.test_labels()
        self.data_max_length+= len(Y)
        idx= np.arange(10000)
        if (shuffle):
            self.tdatarng.shuffle(idx)
        X= X[idx]
        self.X_test_orig= X[:p["N_TEST"],:,:]
        Y= Y[idx]
        self.Y_test_orig= Y[:p["N_TEST"]]

    def reduce_classes(self, X, Y, Z, classes):
        idx= [y in classes for y in Y]
        newX= X[idx]
        newY= Y[idx]
        newZ= Z[idx]
        return (newX, newY, newZ)
    
    def spike_time_from_gray(self,t):
        return (255.0-t)/255.0*(p["TRIAL_MS"]-4*p["DT_MS"])+2*p["DT_MS"]   # make sure spikes are two timesteps within the presentation window

    def spike_time_from_gray2(self,t):
        t= t/255.0*10.0
        return 10.0*np.log(t/(t-0.2))

    """ 
    generate a spikeTimes array and startSpike and endSpike arrays to allow indexing into the 
    spikeTimes in a shuffled way
    """
    # ISSUE: here we are not rounding to the multiples of batch size!
    # When data loading, we are doing that for N_trial ...
    # Needs tidying up!
    def generate_input_spiketimes_shuffle_fast(self, p, Xtrain, Ytrain, Xeval, Yeval):
        if Xtrain is None:
            X= Xeval.reshape(Xeval.shape[0], self.num_input)
            Y= Yeval
            padN= p["N_BATCH"]- ((len(X)-1) % p["N_BATCH"] +1)
            X= np.vstack([X, np.zeros((padN, self.num_input))])
            Y= np.append(Y, [-1]*padN)
        else:
            X= Xtrain.reshape(Xtrain.shape[0], self.num_input)
            Y= Ytrain
            padN= p["N_BATCH"]- ((len(X)-1) % p["N_BATCH"] +1)
            X= np.vstack([X, np.zeros((padN, self.num_input))])
            Y= np.append(Y, [-1]*padN)
            if Xeval is not None:
                X= np.vstack([X, Xeval.reshape(Xeval.shape[0], self.num_input)])    
                Y= np.append(Y, Yeval)
                padN= p["N_BATCH"]- ((len(X)-1) % p["N_BATCH"] +1)
                X= np.vstack([X, np.zeros((padN, self.num_input))])
                Y= np.append(Y, [-1]*padN)
        Y= Y.astype(int)
        # N is the number of training/testing images: always use all images given
        N= len(Y)
        all_sts= []
        all_input_end= []
        all_input_start= []
        stidx_offset= 0
        self.max_stim_time= 0.0
        for i in range(N):
            tx= X[i,:]
            # identify the neurons that are spiking (threshold 1 on graylevels)
            ix= tx > 1
            # select only those neurons that spike to produce a spike time
            tx= tx[ix]
            # generate spike times; note, these are relative to trial start
            tx= self.spike_time_from_gray(tx)
            if len(tx) > 0:
                self.max_stim_time= max(self.max_stim_time, np.amax(tx))
            # determine i_end from ix so that the right spike times are attributed to the
            # right neurons (uses True == 1, False == 0 in cumsum)
            i_end= np.cumsum(ix)+stidx_offset
            all_sts.append(tx)
            i_start= np.empty(i_end.shape)
            i_start[0]= stidx_offset
            i_start[1:]= i_end[:-1]
            all_input_end.append(i_end)
            all_input_start.append(i_start)
            stidx_offset= i_end[-1]
        X= np.hstack(all_sts)
        input_end= np.hstack(all_input_end)
        input_start= np.hstack(all_input_start)
        return (X, Y, input_start, input_end) 
                
    def define_model(self, p, shuffle):
        self.trial_steps= int(round(p["TRIAL_MS"]/p["DT_MS"]))

        # ----------------------------------------------------------------------------
        # Model description
        # ----------------------------------------------------------------------------
        kwargs = {}
        if p["CUDA_VISIBLE_DEVICES"]:
            from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL
            kwargs["selectGPUByDeviceID"] = True
            kwargs["deviceSelectMethod"] = DeviceSelect_MANUAL
        self.model = genn_model.GeNNModel("float", p["NAME"], generateLineInfo=True, time_precision="double", **kwargs)
        self.model.dT = p["DT_MS"]
        self.model.timing_enabled = p["TIMING"]
        self.model.batch_size = p["N_BATCH"]
        if p["MODEL_SEED"] is not None:
            self.model._model.set_seed(p["MODEL_SEED"])


        # ----------------------------------------------------------------------------
        # input neuron initialisation
        # ----------------------------------------------------------------------------

        input_params= {
            "N_neurons": self.num_input,
            "N_max_spike": p["N_MAX_SPIKE"],
        }
        self.input_init_vars= {
            "startSpike": 0.0,  # to be set later
            "endSpike": 0.0,    # to be set later
            "back_spike": 0,
            "rp_ImV": p["N_MAX_SPIKE"]-1,
            "wp_ImV": 0,
            "fwd_start": p["N_MAX_SPIKE"]-1,
            "new_fwd_start": p["N_MAX_SPIKE"]-1,
            "rev_t": 0.0,
        }
        self.input= self.model.add_neuron_population("input", self.num_input, EVP_SSA_MNIST_SHUFFLE, input_params, self.input_init_vars)
        self.input.set_extra_global_param("t_k", -1e5*np.ones(p["N_BATCH"]*self.num_input*p["N_MAX_SPIKE"], dtype=np.float32))
        # reserve enough space for any set of input spikes that is likely
        self.input.set_extra_global_param("spikeTimes", np.zeros(200000000, dtype=np.float32))

        input_reset_params= {"N_max_spike": p["N_MAX_SPIKE"]}
        input_reset_var_refs= {
            "back_spike": genn_model.create_var_ref(self.input, "back_spike"),
            "rp_ImV": genn_model.create_var_ref(self.input, "rp_ImV"),
            "wp_ImV": genn_model.create_var_ref(self.input, "wp_ImV"),
            "fwd_start": genn_model.create_var_ref(self.input, "fwd_start"),
            "new_fwd_start": genn_model.create_var_ref(self.input, "new_fwd_start"),
            "rev_t": genn_model.create_var_ref(self.input, "rev_t"),
        }
        self.input_reset= self.model.add_custom_update("input_reset", "neuronReset", EVP_input_reset_MNIST, input_reset_params, {}, input_reset_var_refs)

        input_set_params= {
            "N_batch": p["N_BATCH"],
            "num_input": self.num_input,
        }
        input_set_var_refs= {
            "startSpike": genn_model.create_var_ref(self.input, "startSpike"),
            "endSpike": genn_model.create_var_ref(self.input, "endSpike"),
        }
        self.input_set= self.model.add_custom_update("input_set", "inputUpdate", EVP_input_set_MNIST_shuffle, input_set_params, {}, input_set_var_refs)
        # reserving memory for the worst case of the full training set
        self.input_set.set_extra_global_param("allStartSpike", np.zeros(self.data_max_length*self.num_input, dtype=int))
        self.input_set.set_extra_global_param("allEndSpike", np.zeros(self.data_max_length*self.num_input, dtype=int))
        self.input_set.set_extra_global_param("allInputID", np.zeros(self.data_max_length, dtype=int))
        self.input_set.set_extra_global_param("trial", 0)


        # ----------------------------------------------------------------------------
        # hidden neuron initialisation
        # ----------------------------------------------------------------------------

        if p["REG_TYPE"] == "none":
            hidden_params= {
                "tau_m": p["TAU_MEM"],
                "V_thresh": p["V_THRESH"],
                "V_reset": p["V_RESET"],
                "N_neurons": p["NUM_HIDDEN"],
                "N_max_spike": p["N_MAX_SPIKE"],
            }
            self.hidden_init_vars= {
                "V": p["V_RESET"],
                "lambda_V": 0.0,
                "lambda_I": 0.0,
                "rev_t": 0.0,
                "rp_ImV": p["N_MAX_SPIKE"]-1,
                "wp_ImV": 0,
                "fwd_start": p["N_MAX_SPIKE"]-1,
                "new_fwd_start": p["N_MAX_SPIKE"]-1,
                "back_spike": 0,
                "tau_syn": p["TAU_SYN"],
            }
            self.hidden= self.model.add_neuron_population("hidden", p["NUM_HIDDEN"], EVP_LIF, hidden_params, self.hidden_init_vars)
            self.hidden.set_extra_global_param("t_k", -1e5*np.ones(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
            self.hidden.set_extra_global_param("ImV", np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))

            hidden_reset_params= {
                "V_reset": p["V_RESET"],
                "N_max_spike": p["N_MAX_SPIKE"],
            }
            hidden_reset_var_refs= {
                "rp_ImV": genn_model.create_var_ref(self.hidden, "rp_ImV"),
                "wp_ImV": genn_model.create_var_ref(self.hidden, "wp_ImV"),
                "V": genn_model.create_var_ref(self.hidden, "V"),
                "lambda_V": genn_model.create_var_ref(self.hidden, "lambda_V"),
                "lambda_I": genn_model.create_var_ref(self.hidden, "lambda_I"),
                "rev_t": genn_model.create_var_ref(self.hidden, "rev_t"),
                "fwd_start": genn_model.create_var_ref(self.hidden, "fwd_start"),
                "new_fwd_start": genn_model.create_var_ref(self.hidden, "new_fwd_start"),
                "back_spike": genn_model.create_var_ref(self.hidden, "back_spike"),
            }
            self.hidden_reset= self.model.add_custom_update("hidden_reset", "neuronReset", EVP_neuron_reset, hidden_reset_params, {}, hidden_reset_var_refs)


        if p["REG_TYPE"] == "simple":
            hidden_params= {
                "tau_m": p["TAU_MEM"],
                "V_thresh": p["V_THRESH"],
                "V_reset": p["V_RESET"],
                "N_neurons": p["NUM_HIDDEN"],
                "N_max_spike": p["N_MAX_SPIKE"],
                "N_batch": p["N_BATCH"],
                "lbd_upper": p["LBD_UPPER"],
                "lbd_lower": p["LBD_LOWER"],
                "nu_upper": p["NU_UPPER"],
            }
            self.hidden_init_vars= {
                "V": p["V_RESET"],
                "lambda_V": 0.0,
                "lambda_I": 0.0,
                "rev_t": 0.0,
                "rp_ImV": p["N_MAX_SPIKE"]-1,
                "wp_ImV": 0,
                "fwd_start": p["N_MAX_SPIKE"]-1,
                "new_fwd_start": p["N_MAX_SPIKE"]-1,
                "back_spike": 0,
                "sNSum": 0.0,
                "new_sNSum": 0.0,
                "tau_syn": p["TAU_SYN"],
           }
            self.hidden= self.model.add_neuron_population("hidden", p["NUM_HIDDEN"], EVP_LIF_reg, hidden_params, self.hidden_init_vars)
            self.hidden.set_extra_global_param("t_k", -1e5*np.ones(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
            self.hidden.set_extra_global_param("ImV", np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
            self.hidden.set_extra_global_param("pDrop", p["PDROP_HIDDEN"])
            
            hidden_reset_params= {
                "V_reset": p["V_RESET"],
                "N_max_spike": p["N_MAX_SPIKE"],
                "N_neurons": p["NUM_HIDDEN"],
            }
            hidden_reset_var_refs= {
                "rp_ImV": genn_model.create_var_ref(self.hidden, "rp_ImV"),
                "wp_ImV": genn_model.create_var_ref(self.hidden, "wp_ImV"),
                "V": genn_model.create_var_ref(self.hidden, "V"),
                "lambda_V": genn_model.create_var_ref(self.hidden, "lambda_V"),
                "lambda_I": genn_model.create_var_ref(self.hidden, "lambda_I"),
                "rev_t": genn_model.create_var_ref(self.hidden, "rev_t"),
                "fwd_start": genn_model.create_var_ref(self.hidden, "fwd_start"),
                "new_fwd_start": genn_model.create_var_ref(self.hidden, "new_fwd_start"),
                "back_spike": genn_model.create_var_ref(self.hidden, "back_spike"),
                "sNSum": genn_model.create_var_ref(self.hidden, "sNSum"),
                "new_sNSum": genn_model.create_var_ref(self.hidden, "new_sNSum"),
            }
            self.hidden_reset= self.model.add_custom_update("hidden_reset", "neuronReset", EVP_neuron_reset_reg, hidden_reset_params, {}, hidden_reset_var_refs)

            if p["AVG_SNSUM"]:
                params= {"reduced_sNSum": 0.0}
                var_refs= {"sNSum": genn_model.create_var_ref(self.hidden, "sNSum")}
                self.hidden_reg_reduce= self.model.add_custom_update("hidden_reg_reduce", "sNSumReduce", EVP_reg_reduce, {}, params, var_refs)

                params= {"N_batch": p["N_BATCH"]}
                var_refs= {
                    "reduced_sNSum": genn_model.create_var_ref(self.hidden_reg_reduce, "reduced_sNSum"),
                    "sNSum": genn_model.create_var_ref(self.hidden, "sNSum")
                }
                self.hidden_redSNSum_apply= self.model.add_custom_update("hidden_redSNSum_apply", "sNSumApply", EVP_sNSum_apply, params, {}, var_refs)


        if p["REG_TYPE"] == "Thomas1":
            hidden_params= {
                "tau_m": p["TAU_MEM"],
                "V_thresh": p["V_THRESH"],
                "V_reset": p["V_RESET"],
                "N_neurons": p["NUM_HIDDEN"],
                "N_max_spike": p["N_MAX_SPIKE"],
                "tau_syn": p["TAU_SYN"],
                "N_batch": p["N_BATCH"],
                "lbd_lower": p["LBD_LOWER"],
                "lbd_upper": p["LBD_UPPER"],
                "nu_upper": p["NU_UPPER"],
                "rho_upper": p["RHO_UPPER"],
                "glb_upper": p["GLB_UPPER"],
                "N_batch": p["N_BATCH"],
            }
            self.hidden_init_vars= {
                "V": p["V_RESET"],
                "lambda_V": 0.0,
                "lambda_I": 0.0,
                "rev_t": 0.0,
                "rp_ImV": p["N_MAX_SPIKE"]-1,
                "wp_ImV": 0,
                "fwd_start": p["N_MAX_SPIKE"]-1,
                "new_fwd_start": p["N_MAX_SPIKE"]-1,
                "back_spike": 0,
                "sNSum": 0.0,
                "new_sNSum": 0.0,
            }
            self.hidden= self.model.add_neuron_population("hidden", p["NUM_HIDDEN"], EVP_LIF_reg_Thomas1, hidden_params, self.hidden_init_vars)
            self.hidden.set_extra_global_param("t_k", -1e5*np.ones(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
            self.hidden.set_extra_global_param("ImV", np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
            self.hidden.set_extra_global_param("sNSum_all", np.zeros(p["N_BATCH"]))

            hidden_reset_params= {
                "V_reset": p["V_RESET"],
                "N_max_spike": p["N_MAX_SPIKE"],
                "N_neurons": p["NUM_HIDDEN"],
            }
            hidden_reset_var_refs= {
                "rp_ImV": genn_model.create_var_ref(self.hidden, "rp_ImV"),
                "wp_ImV": genn_model.create_var_ref(self.hidden, "wp_ImV"),
                "V": genn_model.create_var_ref(self.hidden, "V"),
                "lambda_V": genn_model.create_var_ref(self.hidden, "lambda_V"),
                "lambda_I": genn_model.create_var_ref(self.hidden, "lambda_I"),
                "rev_t": genn_model.create_var_ref(self.hidden, "rev_t"),
                "fwd_start": genn_model.create_var_ref(self.hidden, "fwd_start"),
                "new_fwd_start": genn_model.create_var_ref(self.hidden, "new_fwd_start"),
                "back_spike": genn_model.create_var_ref(self.hidden, "back_spike"),
                "sNSum": genn_model.create_var_ref(self.hidden, "sNSum"),
                "new_sNSum": genn_model.create_var_ref(self.hidden, "new_sNSum"),
            }
            self.hidden_reset= self.model.add_custom_update("hidden_reset", "neuronReset", EVP_neuron_reset_reg_global, hidden_reset_params, {}, hidden_reset_var_refs)
            self.hidden_reset.set_extra_global_param("sNSum_all", np.zeros(p["N_BATCH"]))

            if p["AVG_SNSUM"]:
                params= {"reduced_sNSum": 0.0}
                var_refs= {"sNSum": genn_model.create_var_ref(self.hidden, "sNSum")}
                self.hidden_reg_reduce= self.model.add_custom_update("hidden_reg_reduce", "sNSumReduce", EVP_reg_reduce, {}, params, var_refs)

                params= {"N_batch": p["N_BATCH"]}
                var_refs= {
                    "reduced_sNSum": genn_model.create_var_ref(self.hidden_reg_reduce, "reduced_sNSum"),
                    "sNSum": genn_model.create_var_ref(self.hidden, "sNSum")
                }
                self.hidden_redSNSum_apply= self.model.add_custom_update("hidden_redSNSum_apply", "sNSumApply", EVP_sNSum_apply, params, {}, var_refs)


        # ----------------------------------------------------------------------------
        # output neuron initialisation
        # ----------------------------------------------------------------------------

        if p["LOSS_TYPE"][:11] == "first_spike":
            output_params= {
                "N_batch": p["N_BATCH"],
                "trial_t": p["TRIAL_MS"],
                "V_thresh": p["V_THRESH"],
                "V_reset": p["V_RESET"],
                "N_neurons": self.num_output,
                "N_max_spike": p["N_MAX_SPIKE"],
                "tau0": p["TAU_0"],
                "tau1": p["TAU_1"],
                "alpha": p["ALPHA"],
            }
            self.output_init_vars= {
                "V": p["V_RESET"],
                "lambda_V": 0.0,
                "lambda_I": 0.0,
                "trial": 0,
                "rev_t": 0.0,
                "rp_ImV": 0,
                "wp_ImV": 0,
                "back_spike": 0,
                "first_spike_t": -1e5,
                "new_first_spike_t": -1e5,
                "exp_st": 0.0,
                "expsum": 1.0,
                "tau_m": p["TAU_MEM"],
                "tau_syn": p["TAU_SYN"],
            }
            if p["LOSS_TYPE"] == "first_spike":
                self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_first_spike, output_params, self.output_init_vars)
            if p["LOSS_TYPE"] == "first_spike_exp":
                self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_first_spike_exp, output_params, self.output_init_vars)
            
            self.output.set_extra_global_param("t_k", -1e5*np.ones(p["N_BATCH"]*self.num_output*p["N_MAX_SPIKE"], dtype=np.float32))
            self.output.set_extra_global_param("ImV", np.zeros(p["N_BATCH"]*self.num_output*p["N_MAX_SPIKE"], dtype=np.float32))
            self.output.set_extra_global_param("label", np.zeros(self.data_max_length, dtype=np.float32)) # reserve space for labels

            output_reset_params= {
                "V_reset": p["V_RESET"],
                "N_class": self.N_class,
                "N_max_spike": p["N_MAX_SPIKE"],
                "tau0": p["TAU_0"],
                "tau1": p["TAU_1"],
            }
            output_reset_var_refs= {
                "V": genn_model.create_var_ref(self.output, "V"),
                "lambda_V": genn_model.create_var_ref(self.output, "lambda_V"),
                "lambda_I": genn_model.create_var_ref(self.output, "lambda_I"),
                "trial": genn_model.create_var_ref(self.output, "trial"),
	            "rp_ImV": genn_model.create_var_ref(self.output, "rp_ImV"),
                "wp_ImV": genn_model.create_var_ref(self.output, "wp_ImV"),
                "rev_t": genn_model.create_var_ref(self.output, "rev_t"),
                "back_spike": genn_model.create_var_ref(self.output, "back_spike"),
                "first_spike_t": genn_model.create_var_ref(self.output, "first_spike_t"),
                "new_first_spike_t": genn_model.create_var_ref(self.output, "new_first_spike_t"),
                "exp_st": genn_model.create_var_ref(self.output, "exp_st"),
                "expsum": genn_model.create_var_ref(self.output, "expsum"),
            }
            self.output_reset= self.model.add_custom_update("output_reset", "neuronReset", EVP_neuron_reset_output_MNIST_first_spike, output_reset_params, {}, output_reset_var_refs)


        if p["LOSS_TYPE"] == "max":
            output_params= {
                "N_batch": p["N_BATCH"],
                "trial_t": p["TRIAL_MS"],
            }
            self.output_init_vars= {
                "V": p["V_RESET"],
                "lambda_V": 0.0,
                "lambda_I": 0.0,
                "trial": 0,
                "max_V": p["V_RESET"],
                "new_max_V": p["V_RESET"],
                "max_t": 0.0,
                "new_max_t": 0.0,
                "rev_t": 0.0,
                "expsum": 1.0,
                "exp_V": 1.0,
                "tau_m": p["TAU_MEM"],
                "tau_syn": p["TAU_SYN"],
            }
            self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_max, output_params, self.output_init_vars)
            self.output.set_extra_global_param("label", np.zeros(self.data_max_length, dtype=np.float32)) # reserve space for labels

            output_reset_params= {
                "V_reset": p["V_RESET"],
                "N_class": self.N_class,
            }
            output_var_refs= {
                "V": genn_model.create_var_ref(self.output, "V"),
                "lambda_V": genn_model.create_var_ref(self.output, "lambda_V"),
                "lambda_I": genn_model.create_var_ref(self.output, "lambda_I"),
                "trial": genn_model.create_var_ref(self.output, "trial"),
                "max_V": genn_model.create_var_ref(self.output, "max_V"),
                "new_max_V": genn_model.create_var_ref(self.output, "new_max_V"),
                "max_t": genn_model.create_var_ref(self.output, "max_t"),
                "new_max_t": genn_model.create_var_ref(self.output, "new_max_t"),
                "rev_t": genn_model.create_var_ref(self.output, "rev_t"),
                "expsum": genn_model.create_var_ref(self.output, "expsum"),
                "exp_V": genn_model.create_var_ref(self.output, "exp_V"),
            }
            self.output_reset= self.model.add_custom_update("output_reset", "neuronReset", EVP_neuron_reset_output_MNIST_max, output_reset_params, {}, output_var_refs)


        if p["LOSS_TYPE"] == "sum":
            output_params= {
                "N_batch": p["N_BATCH"],
                "trial_t": p["TRIAL_MS"],
            }
            self.output_init_vars= {
                "V": p["V_RESET"],
                "lambda_V": 0.0,
                "lambda_I": 0.0,
                "trial": 0,
                "sum_V": 0.0,
                "SoftmaxVal": 0.0,
                "rev_t": 0.0,
                "tau_m": p["TAU_MEM"],
                "tau_syn": p["TAU_SYN"],
            }
            self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_sum, output_params, self.output_init_vars)
            self.output.set_extra_global_param("label", np.zeros(self.data_max_length, dtype=np.float32)) # reserve space for labels

            # updates to do do softmax
            softmax_1_init_vars= {
                "MaxVal": -1e10
            }
            softmax_1_var_refs= {
                "Val": genn_model.create_var_ref(self.output, "sum_V")
            }
            softmax_1= self.model.add_custom_update("softmax_1", "Softmax1", softmax_1_model, {}, softmax_1_init_vars, softmax_1_var_refs)
            softmax_2_init_vars= {
                "SumExpVal": 0.0
            }
            softmax_2_var_refs= {
                "Val": genn_model.create_var_ref(self.output, "sum_V"),
                "MaxVal": genn_model.create_var_ref(softmax_1, "MaxVal")
            }
            softmax_2= self.model.add_custom_update("softmax_2", "Softmax2", softmax_2_model, {}, softmax_2_init_vars, softmax_2_var_refs)
            softmax_3_var_refs= {
                "Val": genn_model.create_var_ref(self.output, "sum_V"),
                "MaxVal": genn_model.create_var_ref(softmax_1, "MaxVal"),
                "SumExpVal": genn_model.create_var_ref(softmax_2, "SumExpVal"),
                "SoftmaxVal":genn_model.create_var_ref(self.output,"SoftmaxVal")
            }
            softmax_3= self.model.add_custom_update("softmax_3", "Softmax3", softmax_3_model, {}, {}, softmax_3_var_refs)
            output_reset_params= {
                "V_reset": p["V_RESET"],
                "N_class": self.N_class,
            }

            output_var_refs= {
                "V": genn_model.create_var_ref(self.output, "V"),
                "lambda_V": genn_model.create_var_ref(self.output, "lambda_V"),
                "lambda_I": genn_model.create_var_ref(self.output, "lambda_I"),
                "trial": genn_model.create_var_ref(self.output, "trial"),
                "sum_V": genn_model.create_var_ref(self.output, "sum_V"),
                "rev_t": genn_model.create_var_ref(self.output, "rev_t"),
            }
            self.output_reset= self.model.add_custom_update("output_reset", "neuronReset", EVP_neuron_reset_output_MNIST_sum, output_reset_params, {}, output_var_refs)


        if p["LOSS_TYPE"] == "avg_xentropy":
            output_params= {
                "N_batch": p["N_BATCH"],
                "trial_t": p["TRIAL_MS"],
                "N_neurons": self.num_output,
                "trial_steps": self.trial_steps,
                "N_class": self.N_class,
            }
            self.output_init_vars= {
                "V": p["V_RESET"],
                "lambda_V": 0.0,
                "lambda_I": 0.0,
                "trial": 0,
                "sum_V": 0.0,
                "rp_V": 0,
                "wp_V": 0,
                "loss": 0,
                "tau_m": p["TAU_MEM"],
                "tau_syn": p["TAU_SYN"],
            }
            self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_MNIST_avg_xentropy, output_params, self.output_init_vars)
            self.output.set_extra_global_param("label", np.zeros(self.data_max_length, dtype=np.float32)) # reserve space for labels
            self.output.set_extra_global_param("Vbuf", np.zeros(p["N_BATCH"]*self.num_output*self.trial_steps*2, dtype=np.float32)) # reserve space for voltage buffer

            output_reset_params= {
                "V_reset": p["V_RESET"],
                "N_class": self.N_class,
                "trial_steps": self.trial_steps,
            }
            output_var_refs= {
                "V": genn_model.create_var_ref(self.output, "V"),
                "lambda_V": genn_model.create_var_ref(self.output, "lambda_V"),
                "lambda_I": genn_model.create_var_ref(self.output, "lambda_I"),
                "trial": genn_model.create_var_ref(self.output, "trial"),
                "sum_V": genn_model.create_var_ref(self.output, "sum_V"),
                "rp_V": genn_model.create_var_ref(self.output, "rp_V"),
                "wp_V": genn_model.create_var_ref(self.output, "wp_V"),
                "loss": genn_model.create_var_ref(self.output, "loss"),
            }
            self.output_reset= self.model.add_custom_update("output_reset", "neuronReset", EVP_neuron_reset_output_avg_xentropy, output_reset_params, {}, output_var_refs)


        # ----------------------------------------------------------------------------
        # Optimiser initialisation
        # ----------------------------------------------------------------------------

        adam_params= {
            "beta1": p["ADAM_BETA1"],
            "beta2": p["ADAM_BETA2"],
            "epsilon": p["ADAM_EPS"],
        }
        self.adam_init_vars= {
            "m": 0.0,
            "v": 0.0,
        }


        # ----------------------------------------------------------------------------
        # Synapse initialisation
        # ----------------------------------------------------------------------------

        self.in_to_hid_init_vars= {"dw": 0}
        self.in_to_hid_init_vars["w"]= genn_model.init_var(
            "Normal", {"mean": p["INPUT_HIDDEN_MEAN"], "sd": p["INPUT_HIDDEN_STD"]})

        self.hid_to_out_init_vars= {"dw": 0}
        self.hid_to_out_init_vars["w"]= genn_model.init_var(
            "Normal", {"mean": p["HIDDEN_OUTPUT_MEAN"], "sd": p["HIDDEN_OUTPUT_STD"]})

        if p["RECURRENT"]:
            self.hid_to_hid_init_vars= {"dw": 0}
            self.hid_to_hid_init_vars["w"]= genn_model.init_var(
                "Normal", {"mean": p["HIDDEN_HIDDEN_MEAN"], "sd": p["HIDDEN_HIDDEN_STD"]})

        self.in_to_hid= self.model.add_synapse_population(
            "in_to_hid", "DENSE_INDIVIDUALG", NO_DELAY, self.input, self.hidden, EVP_input_synapse,
            {}, self.in_to_hid_init_vars, {}, {}, my_Exp_Curr, {}, {})

        self.hid_to_out= self.model.add_synapse_population(
            "hid_to_out", "DENSE_INDIVIDUALG", NO_DELAY, self.hidden, self.output, EVP_synapse,
            {}, self.hid_to_out_init_vars, {}, {}, my_Exp_Curr, {}, {})

        if p["RECURRENT"]:
            self.hid_to_hid= self.model.add_synapse_population(
                "hid_to_hid", "DENSE_INDIVIDUALG", NO_DELAY, self.hidden, self.hidden, EVP_synapse,
                {}, self.hid_to_hid_init_vars, {}, {}, my_Exp_Curr, {}, {})

        self.optimisers= []
        var_refs = {"dw": genn_model.create_wu_var_ref(self.in_to_hid, "dw")}
        self.in_to_hid_reduce= self.model.add_custom_update(
            "in_to_hid_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
        var_refs = {"gradient": genn_model.create_wu_var_ref(self.in_to_hid_reduce, "reduced_dw"),
                    "variable": genn_model.create_wu_var_ref(self.in_to_hid, "w")}
        self.in_to_hid_learn= self.model.add_custom_update(
            "in_to_hid_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)
        self.optimisers.append(self.in_to_hid_learn)
        
        var_refs = {"dw": genn_model.create_wu_var_ref(self.hid_to_out, "dw")}
        self.hid_to_out_reduce= self.model.add_custom_update(
            "hid_to_out_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
        var_refs = {"gradient": genn_model.create_wu_var_ref(self.hid_to_out_reduce, "reduced_dw"),
                    "variable": genn_model.create_wu_var_ref(self.hid_to_out, "w")}
        self.hid_to_out_learn= self.model.add_custom_update(
            "hid_to_out_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)
        self.hid_to_out.pre_target_var= "revIsyn"
        self.optimisers.append(self.hid_to_out_learn)

        if p["RECURRENT"]:
            var_refs = {"dw": genn_model.create_wu_var_ref(self.hid_to_hid, "dw")}
            self.hid_to_hid_reduce= self.model.add_custom_update(
                "hid_to_hid_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
            var_refs = {"gradient": genn_model.create_wu_var_ref(self.hid_to_hid_reduce, "reduced_dw"),
                        "variable": genn_model.create_wu_var_ref(self.hid_to_hid, "w")}
            self.hid_to_hid_learn= self.model.add_custom_update(
                "hid_to_hid_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)
            self.hid_to_hid.pre_target_var= "revIsyn"
            self.optimisers.append(self.hid_to_hid_learn)

        # DEBUG hidden layer spike numbers
        if p["DEBUG_HIDDEN_N"]:
            if p["REG_TYPE"] != "Thomas1":
                self.model.neuron_populations["hidden"].spike_recording_enabled= True

        # enable buffered spike recording where desired
        for pop in p["REC_SPIKES"]:
            self.model.neuron_populations[pop].spike_recording_enabled= True

    """
    ----------------------------------------------------------------------------
    Run the model
    ----------------------------------------------------------------------------
    """
            
    def run_model(self, number_epochs, p, shuffle, X_train= None, labels_train= None, X_eval= None, labels_eval= None, resfile= None):
        if p["LOAD_LAST"]:
            self.in_to_hid.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_input_hidden_last.npy"))
            self.in_to_hid.push_var_to_device("w")
            self.hid_to_out.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_output_last.npy"))
            self.hid_to_out.push_var_to_device("w")
            if p["RECURRENT"]:
                self.hid_to_hid.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_hidden_last.npy"))
                self.hid_to_hid.push_var_to_device("w")

        # set up run
        N_trial= 0
        if X_train is not None:
            assert(labels_train is not None)
            N_trial_train= (len(X_train)-1) // p["N_BATCH"] + 1  # large enough to fit potentially incomplete last batch
            N_trial+= N_trial_train
        else:
            N_trial_train= 0
        if X_eval is not None:
            assert(labels_eval is not None)
            N_trial_eval= (len(X_eval)-1) // p["N_BATCH"] + 1  # large enough to fit potentially incomplete last batch
            N_trial+= N_trial_eval
        else:
            N_trial_eval= 0

        print(f"N_trial_train= {N_trial_train}, N_trial_eval= {N_trial_eval}")
        adam_step= 1
        learning_rate= p["ETA"]

        # set up recording if required
        spike_t= {}
        spike_ID= {}
        for pop in p["REC_SPIKES"]:
            spike_t[pop]= []
            spike_ID[pop]= []
        rec_spk_lbl= []
        rec_spk_pred= []
        rec_vars_n= {}
        for pop, var in p["REC_NEURONS"]:
            rec_vars_n[var+pop]= []
        #rec_exp_V= []
        #rec_expsum= []
        rec_n_t= []
        rec_n_lbl= []
        rec_n_pred= []
        rec_vars_s= {}
        for pop, var in p["REC_SYNAPSES"]:
            rec_vars_s[var+pop]= []
        rec_s_t= []
        rec_s_lbl= []
        rec_s_pred= []

        # build and assign the input spike train and corresponding labels
        # these are padded to multiple of batch size for both train and eval portions
        X, Y, input_start, input_end= self.generate_input_spiketimes_shuffle_fast(p, X_train, labels_train, X_eval, labels_eval)
        self.input.extra_global_params["spikeTimes"].view[:len(X)]= X
        self.input.push_extra_global_param_to_device("spikeTimes")
        self.input_set.extra_global_params["allStartSpike"].view[:len(input_start)]= input_start
        self.input_set.push_extra_global_param_to_device("allStartSpike")
        self.input_set.extra_global_params["allEndSpike"].view[:len(input_end)]= input_end
        self.input_set.push_extra_global_param_to_device("allEndSpike")
        if labels_train is not None:
            input_id= np.arange(labels_train.shape[0])
        else:
            input_id= []
        all_input_id= np.arange(Y.shape[0])
        self.input_set.extra_global_params["allInputID"].view[:len(all_input_id)]= all_input_id
        self.input_set.push_extra_global_param_to_device("allInputID")

        for epoch in range(number_epochs):
            if N_trial_train > 0 and shuffle:
                # by virtue of input_id being the right length we do not shuffle
                # padding inputs
                self.datarng.shuffle(input_id)
                all_input_id[:len(input_id)]= input_id
                Y[:len(input_id)]= labels_train[input_id]
                self.output.extra_global_params["label"].view[:len(Y)]= Y
                self.output.push_extra_global_param_to_device("label")
                self.input_set.extra_global_params["allInputID"].view[:len(all_input_id)]= all_input_id
                self.input_set.push_extra_global_param_to_device("allInputID")

            predict= {
                "train": [],
                "eval": []
            }
            the_loss= {
                "train": [],
                "eval": []
            }
            good= {
                "train": 0.0,
                "eval": 0.0
            }

            self.model.t= 0.0
            self.model.timestep= 0
            for var, val in self.input_init_vars.items():
                self.input.vars[var].view[:]= val
            self.input.push_state_to_device()
            self.input.extra_global_params["pDrop"].view[:]= p["PDROP_INPUT"]
            for var, val in self.hidden_init_vars.items():
                self.hidden.vars[var].view[:]= val
            self.hidden.push_state_to_device()
            self.hidden.extra_global_params["pDrop"].view[:]= p["PDROP_HIDDEN"] 
            for var, val in self.output_init_vars.items():
                self.output.vars[var].view[:]= val
            self.output.push_state_to_device()
            self.model.custom_update("EVPReduce")  # this zeros dw (so as to ignore eval gradients from last epoch!)

            if p["DEBUG_HIDDEN_N"]:
                all_hidden_n= []
                all_sNSum= []

            spk_rec_offset= 0
            for trial in range(N_trial):
                trial_end= (trial+1)*p["TRIAL_MS"]

                # assign the input spike train and corresponding labels
                if trial < N_trial_train:
                    phase= "train"
                    # set the actual batch size
                    if trial == N_trial_train-1:
                        N_batch= len(X_train)-(N_trial_train-1)*p["N_BATCH"]
                    else:
                        N_batch= p["N_BATCH"]
                else:
                    phase= "eval"
                    # set the actual batch size
                    if trial == N_trial_train+N_trial_eval-1:
                        N_batch= len(X_eval)-(N_trial_eval-1)*p["N_BATCH"]
                    else:
                        N_batch= p["N_BATCH"]
                    self.input.extra_global_params["pDrop"].view[:]= 0.0
                    self.hidden.extra_global_params["pDrop"].view[:]= 0.0
                self.input_set.extra_global_params["trial"].view[:]= trial
                self.model.custom_update("inputUpdate")
                self.input.extra_global_params["t_offset"].view[:]= self.model.t

                if p["DEBUG_HIDDEN_N"]:
                    if p["REG_TYPE"] != "Thomas1":
                        spike_N_hidden= np.zeros(N_batch)

                int_t= 0
                while (self.model.t < trial_end-1e-1*p["DT_MS"]):
                    self.model.step_time()
                    int_t += 1

                    # DEBUG of middle layer activity
                    if p["DEBUG_HIDDEN_N"]:
                        if int_t%p["SPK_REC_STEPS"] == 0:
                            if p["REG_TYPE"] != "Thomas1":
                                self.model.pull_recording_buffers_from_device()
                                x= self.model.neuron_populations["hidden"].spike_recording_data
                                for btch in range(N_batch):
                                    spike_N_hidden[btch]+= len(x[btch][0])

                    if len(p["REC_SPIKES"]) > 0:
                        if int_t%p["SPK_REC_STEPS"] == 0:
                            if [epoch,trial] in p["REC_SPIKES_EPOCH_TRIAL"]:
                                self.model.pull_recording_buffers_from_device()
                                for pop in p["REC_SPIKES"]:
                                    the_pop= self.model.neuron_populations[pop]
                                    x= the_pop.spike_recording_data
                                    if p["N_BATCH"] > 1:
                                        for i in range(N_batch):
                                            spike_t[pop].append(x[i][0]+(spk_rec_offset+i)*p["TRIAL_MS"])
                                            spike_ID[pop].append(x[i][1])
                                    else:
                                        spike_t[pop].append(x[0]+spk_rec_offset*p["TRIAL_MS"])
                                        spike_ID[pop].append(x[1])
                    if ([epoch,trial] in p["REC_NEURONS_EPOCH_TRIAL"]):
                        for pop, var in p["REC_NEURONS"]:
                            the_pop= self.model.neuron_populations[pop]
                            the_pop.pull_var_from_device(var)
                            rec_vars_n[var+pop].append(the_pop.vars[var].view.copy())
                        rec_n_t.append(self.model.t)
                            
                    if ([epoch,trial] in p["REC_SYNAPSES_EPOCH_TRIAL"]):
                        for pop, var in p["REC_SYNAPSES"]:
                            the_pop= self.model.synapse_populations[pop]
                            if var == "in_syn":
                                the_pop.pull_in_syn_from_device()
                                rec_vars_s[var+pop].append(the_pop.in_syn.copy())
                            else:
                                the_pop.pull_var_from_device(var)
                                rec_vars_s[var+pop].append(the_pop.vars[var].view.copy())
                        rec_s_t.append(self.model.t)

                    # clamp in_syn to 0 one timestep before trial end to avoid bleeding spikes into the next trial
                    if np.abs(self.model.t + p["DT_MS"] - trial_end) < 1e-1*p["DT_MS"]:
                        self.in_to_hid.in_syn[:]= 0.0
                        self.in_to_hid.push_in_syn_to_device()
                        self.hid_to_out.in_syn[:]= 0.0
                        self.hid_to_out.push_in_syn_to_device()

                # at end of trial
                spk_rec_offset+= N_batch-1  # the -1 for compensating the normal progression of time

                # do not learn after the 0th trial where lambdas are meaningless
                if (phase == "train") and trial > 0:
                    update_adam(learning_rate, adam_step, self.optimisers)
                    adam_step += 1
                    self.model.custom_update("EVPReduce")
                    #if trial%2 == 1:
                    self.model.custom_update("EVPLearn")

                self.in_to_hid.in_syn[:]= 0.0
                self.in_to_hid.push_in_syn_to_device()
                self.hid_to_out.in_syn[:]= 0.0
                self.hid_to_out.push_in_syn_to_device()

                if p["REG_TYPE"] == "Thomas1":
                    # for hidden regularistation prepare "sNSum_all"
                    self.hidden_reset.extra_global_params["sNSum_all"].view[:]= np.zeros(p["N_BATCH"])
                    self.hidden_reset.push_extra_global_param_to_device("sNSum_all")

                if p["LOSS_TYPE"][:11]  == "first_spike":
                    # need to copy new_first_spike_t from device before neuronReset!
                    self.output.pull_var_from_device("new_first_spike_t")
                    nfst= self.output.vars["new_first_spike_t"].view[:N_batch,:self.N_class].copy()
                    # neurons that did not spike set to spike time in the future
                    nfst[nfst < 0.0]= self.model.t + p["TRIAL_MS"]
                    pred= np.argmin(nfst, axis=-1)

                if p["LOSS_TYPE"][:3] == "sum":
                    # do the custom updates for softmax!
                    self.model.custom_update("Softmax1")
                    self.model.custom_update("Softmax2")
                    self.model.custom_update("Softmax3")
                    #self.output.pull_var_from_device("sum_V")

                if p["LOSS_TYPE"] == "avg_xentropy":
                    # need to copy sum_V and loss from device before neuronReset!
                    self.output.pull_var_from_device("sum_V")
                    self.output.pull_var_from_device("loss")

                self.model.custom_update("neuronReset")

                if (p["REG_TYPE"] == "simple" or p["REG_TYPE"] == "Thomas1") and p["AVG_SNSUM"]:
                    self.model.custom_update("sNSumReduce")
                    self.model.custom_update("sNSumApply")

                if p["REG_TYPE"] == "Thomas1": 
                    self.hidden_reset.pull_extra_global_param_from_device("sNSum_all")
                    self.hidden.extra_global_params["sNSum_all"].view[:]= self.hidden_reset.extra_global_params["sNSum_all"].view[:]
                    self.hidden.push_extra_global_param_to_device("sNSum_all")
                    if p["DEBUG_HIDDEN_N"]:
                        spike_N_hidden= self.hidden_reset.extra_global_params["sNSum_all"].view[:N_batch].copy()

                # record training loss and error
                # NOTE: the neuronReset does the calculation of expsum and updates exp_V for loss types sum and max
                if p["LOSS_TYPE"] == "max":
                    self.output.pull_var_from_device("exp_V")
                    pred= np.argmax(self.output.vars["exp_V"].view[:N_batch,:self.N_class], axis=-1)
                    
                if p["LOSS_TYPE"] == "sum":
                     self.output.pull_var_from_device("SoftmaxVal")
                     pred= np.argmax(self.output.vars["SoftmaxVal"].view[:N_batch,:self.N_class], axis=-1)
                     
                if p["LOSS_TYPE"] == "avg_xentropy":
                    pred= np.argmax(self.output.vars["sum_V"].view[:N_batch,:self.N_class], axis=-1)
                    
                lbl= Y[(trial*p["N_BATCH"]):(trial*p["N_BATCH"]+N_batch)]

                if ([epoch, trial] in p["REC_SPIKES_EPOCH_TRIAL"]):
                    rec_spk_lbl.append(lbl.copy())
                    rec_spk_pred.append(pred.copy())
                if ([epoch, trial] in p["REC_NEURONS_EPOCH_TRIAL"]):
                    rec_n_lbl.append(lbl.copy())
                    rec_n_pred.append(pred.copy())
                if ([epoch, trial] in p["REC_SYNAPSES_EPOCH_TRIAL"]):
                    rec_s_lbl.append(lbl.copy())
                    rec_s_pred.append(pred.copy())

                if p["LOSS_TYPE"][:11] == "first_spike":
                    self.output.pull_var_from_device("expsum")
                    self.output.pull_var_from_device("exp_st")
                    if p["LOSS_TYPE"] == "first_spike":
                        losses= self.loss_func_first_spike(nfst, lbl, trial, self.N_class, N_batch)
                    if p["LOSS_TYPE"] == "first_spike_exp":
                        losses= self.loss_func_first_spike_exp(nfst, lbl, trial, self.N_class, N_batch)

                if p["LOSS_TYPE"] == "max":
                    self.output.pull_var_from_device("expsum")
                    losses= self.loss_func_max(lbl, N_batch)   # uses self.output.vars["exp_V"].view and self.output.vars["expsum"].view

                if p["LOSS_TYPE"] == "sum":
                    losses= self.loss_func_sum(lbl, N_batch)   # uses self.output.vars["SoftmaxVal"].view 
                    
                if p["LOSS_TYPE"] == "avg_xentropy":
                    losses= self.loss_func_avg_xentropy(N_batch)   # uses self.output.vars["loss"].view

                good[phase] += np.sum(pred == lbl)
                predict[phase].append(pred)
                the_loss[phase].append(losses)

                if p["DEBUG_HIDDEN_N"]:
                    all_hidden_n.append(spike_N_hidden)
                    self.hidden.pull_var_from_device("sNSum")
                    all_sNSum.append(np.mean(self.hidden.vars['sNSum'].view[:N_batch],axis= 0))

                if ([epoch, trial] in p["W_OUTPUT_EPOCH_TRIAL"]): 
                    self.in_to_hid.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_input_hidden_e{}_t{}.npy".format(epoch,trial)), self.in_to_hid.vars["w"].view.copy())
                    self.hid_to_out.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_output_e{}_t{}.npy".format(epoch,trial)), self.hid_to_out.vars["w"].view.copy())
                    if p["RECURRENT"]:
                        self.hid_to_hid.pull_var_from_device("w")
                        np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_hidden_e{}_t{}.npy".format(epoch,trial)), self.hid_to_hid.vars["w"].view.copy())

            if N_trial_train > 0:
                correct= good["train"]/len(X_train)
            else:
                correct= 0
            if N_trial_eval > 0:
                correct_eval= good["eval"]/len(X_eval)
            else:
                correct_eval= 0

            if p["DEBUG_HIDDEN_N"]:
                all_hidden_n= np.hstack(all_hidden_n)
                all_sNSum= np.hstack(all_sNSum)
                print("Hidden spikes in model per trial: {} +/- {}, min {}, max {}".format(np.mean(all_hidden_n),np.std(all_hidden_n),np.amin(all_hidden_n),np.amax(all_hidden_n)))
                print("Hidden spikes per trial per neuron across batches: {} +/- {}, min {}, max {}".format(np.mean(all_sNSum),np.std(all_sNSum),np.amin(all_sNSum),np.amax(all_sNSum)))

            print("{} Training Correct: {}, Training Loss: {}, Evaluation Correct: {}, Evaluation Loss: {}".format(epoch, correct, np.mean(the_loss["train"]), correct_eval, np.mean(the_loss["eval"])))

            if resfile is not None:
                resfile.write("{} {} {} {} {}".format(epoch, correct, np.mean(the_loss["train"]), correct_eval, np.mean(the_loss["eval"])))
                if p["DEBUG_HIDDEN_N"]:
                    resfile.write(" {} {} {} {}".format(np.mean(all_hidden_n),np.std(all_hidden_n),np.amin(all_hidden_n),np.amax(all_hidden_n)))
                    resfile.write(" {} {} {} {}".format(np.mean(all_sNSum),np.std(all_sNSum),np.amin(all_sNSum),np.amax(all_sNSum)))
                resfile.write("\n")
                resfile.flush()
            predict[phase]= np.hstack(predict[phase])

        for pop in p["REC_SPIKES"]:
            spike_t[pop]= np.hstack(spike_t[pop])
            spike_ID[pop]= np.hstack(spike_ID[pop])

        for rec_t, rec_var, rec_list in [ (rec_n_t, rec_vars_n, p["REC_NEURONS"]), (rec_s_t, rec_vars_s, p["REC_SYNAPSES"])]:
            for pop, var in rec_list:
                rec_var[var+pop]= np.array(rec_var[var+pop])
            rec_t= np.array(rec_t)

        rec_spk_lbl= np.array(rec_spk_lbl)
        rec_spk_pred= np.array(rec_spk_pred)
        rec_n_lbl= np.array(rec_n_lbl)
        rec_n_pred= np.array(rec_n_pred)
        rec_s_lbl= np.array(rec_s_lbl)
        rec_s_pred= np.array(rec_s_pred)

        # Saving results
        if p["WRITE_TO_DISK"]:
            if len(p["REC_SPIKES"]) > 0:
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_spk_lbl"), rec_spk_lbl)
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_spk_pred"), rec_spk_pred)
                for pop in p["REC_SPIKES"]:
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_"+pop+"_spike_t"), spike_t[pop])
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_"+pop+"_spike_ID"), spike_ID[pop])

            if len(p["REC_NEURONS"]) > 0:
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_n_t"), rec_n_t)
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_n_lbl"), rec_n_lbl)
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_n_pred"), rec_n_pred)
                for pop, var in p["REC_NEURONS"]:
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_"+var+pop), rec_vars_n[var+pop])

            if len(p["REC_SYNAPSES"]) > 0:
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_t"), rec_s_t)
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_s_lbl"), rec_s_lbl)
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_s_pred"), rec_s_pred)
                for pop, var in p["REC_SYNAPSES"]:
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_"+var+pop), rec_vars_s[var+pop])

        # Saving results
        if p["WRITE_TO_DISK"]:
            self.in_to_hid.pull_var_from_device("w")
            np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_input_hidden_last.npy"), self.in_to_hid.vars["w"].view)
            self.hid_to_out.pull_var_from_device("w")
            np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_output_last.npy"), self.hid_to_out.vars["w"].view)
            if p["RECURRENT"]:
                self.hid_to_hid.pull_var_from_device("w")
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_hidden_last.npy"), self.hid_to_hid.vars["w"].view)

        if p["TIMING"]:
            print("Init: %f" % self.model.init_time)
            print("Init sparse: %f" % self.model.init_sparse_time)
            print("Neuron update: %f" % self.model.neuron_update_time)
            print("Presynaptic update: %f" % self.model.presynaptic_update_time)
            print("Synapse dynamics: %f" % self.model.synapse_dynamics_time)
            print("Neuron reset: %f" % self.model.get_custom_update_time("neuronReset"))
            if (p["REG_TYPE"] == "simple" or p["REG_TYPE"] == "Thomas1") and p["AVG_SNSUM"]:
                print("sNSumReduce: %f" % self.model.get_custom_update_time("sNSumReduce"))
                print("sNSumApply: %f" % self.model.get_custom_update_time("sNSumApply"))
            print("EVPReduce: %f" % self.model.get_custom_update_time("EVPReduce"))
            print("EVPLearn: %f" % self.model.get_custom_update_time("EVPLearn"))
            print("inputUpdate: %f" % self.model.get_custom_update_time("inputUpdate"))

        return (spike_t, spike_ID, rec_vars_n, rec_vars_s, correct, correct_eval)

    def train(self, p):
        self.define_model(p, p["SHUFFLE"])
        if p["BUILD"]:
            self.model.build()
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "a")
        return self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_train= self.X_train_orig, labels_train= self.Y_train_orig, X_eval= self.X_val_orig, labels_eval= self.Y_val_orig, resfile= resfile)

    def test(self, p):
        self.define_model(p, False)
        if p["BUILD"]:
            self.model.build()
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        return self.run_model(1, p, False, X_eval= self.X_test_orig, labels_eval= self.Y_test_orig)

    def train_test(self, p):
        # for this, p["N_TRAIN"] should be the full 60000
        self.define_model(p, p["SHUFFLE"])
        if p["BUILD"]:
            self.model.build()
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "a")
        return self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_train= self.X_train_orig, labels_train= self.Y_train_orig, X_eval= self.X_test_orig, labels_eval= self.Y_test_orig, resfile= resfile)
        
