import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from utils import random_shift, random_dilate, ID_jitter
import mnist
#import tonic
from models import *
import os
import urllib.request
import gzip, shutil
from tensorflow.keras.utils import get_file
import tables
import copy
from time import perf_counter
#from enose_data_loader import enose_data_load

#from src.data_loader import EnoseDataLoader
# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
p= {}
p["NAME"]= "test"
p["DATASET"] = None
p["DEBUG"]= False
p["DEBUG_HIDDEN_N"]= False
p["OUT_DIR"]= "."
p["DT_MS"] = 0.1
p["BUILD"] = True
p["TIMING"] = True
p["TRAIN_DATA_SEED"]= 123
p["TEST_DATA_SEED"]= 456
p["MODEL_SEED"]= None

# Experiment parameters
p["TRIAL_MS"]= 20.0
p["N_MAX_SPIKE"]= 400    # make buffers for maximally 400 spikes (200 in a 30 ms trial) - should be safe
p["N_BATCH"]= 32
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 55000
p["N_VALIDATE"]= 5000
p["N_EPOCH"]= 10
p["SHUFFLE"]= True
p["N_TEST"]= 10000
# Network structure
p["NUM_HIDDEN"] = 350

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
p["LBD_UPPER"]= 0.000005
p["LBD_LOWER"]= 0.001
p["NU_UPPER"]= 20*p["N_BATCH"]
p["NU_LOWER"]= 0.1*p["N_BATCH"]
p["RHO_UPPER"]= 5000.0
p["GLB_UPPER"]= 0.00001
# Learning parameters
p["ETA"]= 5e-3
p["ADAM_BETA1"]= 0.9      
p["ADAM_BETA2"]= 0.999    
p["ADAM_EPS"]= 1e-8       
# applied every epoch
p["ETA_DECAY"]= 0.95
# try a step-down of learning rate after substantial training
p["ETA_FIDDELING"]= False
p["ETA_REDUCE"]= 0.1
p["ETA_REDUCE_PERIOD"]= 50

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
# possible loss types: "max", "sum", "avg_xentropy"
p["LOSS_TYPE"]= "max"
p["EVALUATION"]= "random"
p["CUDA_VISIBLE_DEVICES"]= False
p["AVG_SNSUM"]= False
p["REDUCED_CLASSES"]= None
p["AUGMENTATION"]= {}
p["DOWNLOAD_SHD"]= False

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

rng= np.random.default_rng()

def update_adam(learning_rate, adam_step, optimiser_custom_updates):
    first_moment_scale = 1.0 / (1.0 - (p["ADAM_BETA1"] ** adam_step))
    second_moment_scale = 1.0 / (1.0 - (p["ADAM_BETA2"] ** adam_step))
    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["alpha"].view[:] = learning_rate
        o.extra_global_params["firstMomentScale"].view[:] = first_moment_scale
        o.extra_global_params["secondMomentScale"].view[:] = second_moment_scale

class SHD_model:
    def __init__(self, p):
        if p["TRAIN_DATA_SEED"] is not None:
            self.datarng= np.random.default_rng(p["TRAIN_DATA_SEED"])
        else:
            self.datarng= np.random.default_rng()        
        if p["TEST_DATA_SEED"] is not None:
            self.tdatarng= np.random.default_rng(p["TEST_DATA_SEED"])
        else:
            self.tdatarng= np.random.default_rng()        
            
        if p["DATASET"] == "SHD":
            self.load_data_SHD_Zenke(p)

        if p["DATASET"] == "enose":
            self.load_data_enose(p)

        if p["REDUCED_CLASSES"] is not None:
            self.X_train_orig, self.Y_train_orig, self.Z_train_orig= self.reduce_classes(self.X_train_orig, self.Y_train_orig, self.Z_train_orig, p["REDUCED_CLASSES"])
            self.X_test_orig, self.Y_test_orig, self.Z_test_orig= self.reduce_classes(self.X_test_orig, self.Y_test_orig, self.Z_test_orig, p["REDUCED_CLASSES"])
            
    def loss_func(self, Y, p):
        expsum= self.output.vars["expsum"].view
        exp_V= self.output.vars["exp_V"].view
        exp_V_correct= np.array([ exp_V[i,y] for i, y in enumerate(Y) ])
        if (np.sum(exp_V_correct == 0) > 0):
            print("exp_V flushed to 0 exception!")
            print(exp_V_correct)
            print(exp_V[np.where(exp_V_correct == 0),:])
            exp_V_correct[exp_V_correct == 0]+= 2e-45 # make sure all exp_V are > 0
            
        loss= -np.sum(np.log(exp_V_correct)-np.log(expsum[:,0]))/p["N_BATCH"]
        return loss

    def loss_func_avg_xentropy(self, Y, p):
        loss= self.output.vars["loss"].view
        loss= np.sum(np.sum(loss))
        return loss
    
    """
    For now I will disable this - uses tonic,which might be nicer but doesn't give access to speaker info
    def load_data_SHD(self, p, shuffle= True):
        if p["TRAIN_DATA_SEED"] is not None:
            self.datarng= np.random.default_rng(p["TRAIN_DATA_SEED"])
        else:
            self.datarng= np.random.default_rng()        
        if p["TEST_DATA_SEED"] is not None:
            self.tdatarng= np.random.default_rng(p["TEST_DATA_SEED"])
        else:
            self.tdatarng= np.random.default_rng()        
        dataset = tonic.datasets.SHD(save_to='./data', train=True)
        sensor_size = dataset.sensor_size
        self.data_full_length= len(dataset)
        self.N_class= len(dataset.classes)
        self.num_input= int(np.product(sensor_size))
        self.num_output= 32   # first power of two greater than class number
        idx= np.arange(self.data_full_length)
        if (shuffle):
            self.datarng.shuffle(idx)

        train_idx= idx[np.arange(p["N_TRAIN"])]
        eval_idx= idx[np.arange(p["N_VALIDATE"])+(self.data_full_length-p["N_VALIDATE"])]
        self.Y_train_orig= np.empty(len(train_idx), dtype= int)
        self.X_train_orig= []
        for i, s in enumerate(train_idx):
            events, label = dataset[s]
            self.Y_train_orig[i]= label
            self.X_train_orig.append(events)
        self.Y_val_orig= np.empty(len(eval_idx), dtype= int)
        self.X_val_orig= []
        for i, s in enumerate(eval_idx):
            events, label = dataset[s]
            self.Y_val_orig[i]= label
            self.X_val_orig.append(events)
        dataset = tonic.datasets.SHD(save_to='./data', train=False)
        self.data_full_length= max(self.data_full_length, len(dataset))
        self.Y_test_orig= np.empty(len(dataset), dtype= int)
        self.X_test_orig= []
        for i in range(len(dataset)):
            events, label = dataset[i]
            self.Y_test_orig[i]= label
            self.X_test_orig.append(events)
    """

    def load_data_SHD_Zenke(self, p):
        cache_dir=os.path.expanduser("~/data")
        cache_subdir="SHD"
        print("Using cache dir: %s"%cache_dir)
        self.num_input= 700
        self.num_output= 32   # first power of two greater than class number
        self.data_full_length= 0
        if p["DOWNLOAD_SHD"]:
            # dowload the SHD data from the Zenke website
            # The remote directory with the data files
            base_url = "https://zenkelab.org/datasets"
            # Retrieve MD5 hashes from remote
            response = urllib.request.urlopen("%s/md5sums.txt"%base_url)
            data = response.read() 
            lines = data.decode('utf-8').split("\n")
            file_hashes = { line.split()[1]:line.split()[0] for line in lines if len(line.split())==2 }
 
            def get_and_gunzip(origin, filename, md5hash=None):
                gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
                hdf5_file_path=gz_file_path[:-3]
                if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
                    print("Decompressing %s"%gz_file_path)
                    with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return hdf5_file_path
            # Download the Spiking Heidelberg Digits (SHD) dataset
            files = [ "shd_train.h5.gz", 
                      "shd_test.h5.gz",
            ]
            hdf5_file_path= []
            fn= files[0]
            origin= "%s/%s"%(base_url,fn)
            hdf5_file_path= get_and_gunzip(origin, fn, md5hash=file_hashes[fn])
        else:
            # use data from local cache
            hdf5_file_path= 'data/SHD/shd_train.h5'
        fileh= tables.open_file(hdf5_file_path, mode='r')
        units= fileh.root.spikes.units
        times= fileh.root.spikes.times
        self.Y_train_orig= fileh.root.labels
        self.Z_train_orig= fileh.root.extra.speaker
        self.data_full_length= max(self.data_full_length, len(units))
        self.N_class= len(set(self.Y_train_orig))
        self.X_train_orig= []
        for i in range(len(units)):
            self.X_train_orig.append({"x": units[i], "t": times[i]})
        self.X_train_orig= np.array(self.X_train_orig)
        # do the test files
        if p["DOWNLOAD_SHD"]:
            # download data from the Zenke website
            fn= files[1]
            origin= "%s/%s"%(base_url,fn)
            hdf5_file_path= get_and_gunzip(origin, fn, md5hash=file_hashes[fn])
        else:
            # use data from local cache
            hdf5_file_path= 'data/SHD/shd_test.h5'
        fileh= tables.open_file(hdf5_file_path, mode='r')
        units= fileh.root.spikes.units
        times= fileh.root.spikes.times
        self.Y_test_orig= fileh.root.labels
        self.Z_test_orig= fileh.root.extra.speaker
        self.data_full_length= max(self.data_full_length, len(units))
        self.X_test_orig= []
        for i in range(len(units)):
            self.X_test_orig.append({"x": units[i], "t": times[i]})
        self.X_test_orig= np.array(self.X_test_orig)

    # def load_data_enose(self,p):
    #     self.num_input= 8
    #     self.num_output= 8   # first power of two greater than class number
    #     self.N_class= 5
    #     self.X_train_orig, self.Y_train_orig, self.X_test_orig, self.Y_test_orig= enose_data_load()
    #     self.data_full_length= len(self.Y_train_orig)
    

    def reduce_classes(self, X, Y, Z, classes):
        idx= [y in classes for y in Y]
        newX= X[idx]
        newY= Y[idx]
        newZ= Z[idx]
        return (newX, newY, newZ)
    
    def split_SHD_random(self, X, Y, p, shuffle= True):
        idx= np.arange(len(X),dtype= int)
        if (shuffle):
            self.datarng.shuffle(idx)
        train_idx= idx[np.arange(p["N_TRAIN"])]
        eval_idx= idx[np.arange(p["N_VALIDATE"])+(self.data_full_length-p["N_VALIDATE"])]
        print(train_idx)
        newX_t= X[train_idx]
        newX_e= X[eval_idx]
        newY_t= Y[train_idx]
        newY_e= Y[eval_idx]
        print(newX_t[0])
        return (newX_t, newY_t, newX_e, newY_e)

    # split off one speaker to form evaluation set
    def split_SHD_speaker(self, X, Y, Z, speaker, p, shuffle= True):
        speaker= np.array(speaker)
        newX_t= X[Z != speaker]
        newY_t= Y[Z != speaker]
        train_idx= np.arange(len(newY_t))
        if shuffle:
            self.datarng.shuffle(train_idx)
        train_idx= train_idx[:p["N_TRAIN"]]
        newX_t= newX_t[train_idx]
        newY_t= newY_t[train_idx]
        newX_e= X[Z == speaker]
        newY_e= Y[Z == speaker]
        eval_idx= np.arange(len(newY_e))
        if shuffle:
            self.datarng.shuffle(eval_idx)
        eval_idx= eval_idx[:p["N_VALIDATE"]]
        newX_e= newX_e[eval_idx]
        newY_e= newY_e[eval_idx]
        return (newX_t, newY_t, newX_e, newY_e)
    
    """ 
    generate a spikeTimes array and startSpike and endSpike arrays to allow indexing into the 
    spikeTimes in a shuffled way
    """
    # ISSUE: here we are not rounding to the multiples of batch size!
    # When data loading, we are doing that for N_trial ...
    # Needs tidying up!
    def generate_input_spiketimes_shuffle_fast(self, p, Xtrain, Ytrain, Xeval, Yeval):
        # N is the number of training/testing images: always use all images given
        if Xtrain is None:
            X= Xeval
            Y= Yeval
        else:
            if Xeval is None:
                X= Xtrain
                Y= Ytrain
            else:
                X= np.append(Xtrain, Xeval, axis= 0)    
                Y= np.append(Ytrain, Yeval, axis= 0)
        N= len(Y)
        all_sts= []
        all_input_end= []
        all_input_start= []
        stidx_offset= 0
        self.max_stim_time= 0.0
        for i in range(N):
            if p["DATASET"] == "MNIST":
                X= np.reshape(X,(N, self.num_input))
                tx= X[i,:]
                ix= tx > 1
                #ix= tx[tx > 5.1]
                tx= tx[ix]
                tx= self.spike_time_from_gray(tx)
                self.max_stim_time= max(self.max_stim_time, np.amax(tx))
                #tx= self.spike_time_from_gray2(tx)
                i_end= np.cumsum(ix)+stidx_offset
            if p["DATASET"] == "SHD" or p["DATASET"] == "enose":
                events= X[i]
                spike_event_ids = events["x"]
                i_end = np.cumsum(np.bincount(spike_event_ids.astype(int), 
                                              minlength=self.num_input))+stidx_offset    
                assert len(i_end) == self.num_input
                tx = events["t"][np.lexsort((events["t"], spike_event_ids))].astype(float)
                if p["DATASET"] == "SHD":
                    tx *= 1000.0
                self.max_stim_time= max(self.max_stim_time, np.amax(tx))
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
        input_params= {"N_neurons": self.num_input,
                       "N_max_spike": p["N_MAX_SPIKE"] 
        }
        self.input_init_vars= {"startSpike": 0.0,  # to be set later
                               "endSpike": 0.0,         # to be set later
                               "back_spike": 0,
                               "rp_ImV": p["N_MAX_SPIKE"]-1,
                               "wp_ImV": 0,
                               "fwd_start": p["N_MAX_SPIKE"]-1,
                               "new_fwd_start": p["N_MAX_SPIKE"]-1,
                               "rev_t": 0.0}
        hidden_params= {"tau_m": p["TAU_MEM"],
                        "V_thresh": p["V_THRESH"],
                        "V_reset": p["V_RESET"],
                        "N_neurons": p["NUM_HIDDEN"],
                        "N_max_spike": p["N_MAX_SPIKE"],
                        "tau_syn": p["TAU_SYN"],
        }
        self.hidden_init_vars= {"V": p["V_RESET"],
                                "lambda_V": 0.0,
                                "lambda_I": 0.0,
                                "rev_t": 0.0,
                                "rp_ImV": p["N_MAX_SPIKE"]-1,
                                "wp_ImV": 0,
                                "fwd_start": p["N_MAX_SPIKE"]-1,
                                "new_fwd_start": p["N_MAX_SPIKE"]-1,
                                "back_spike": 0,
        }
        output_params= {"tau_m": p["TAU_MEM"],
                        "tau_syn": p["TAU_SYN"],
                        "trial_t": p["TRIAL_MS"],
        }

        output_params["N_batch"]= p["N_BATCH"]

        if p["LOSS_TYPE"] == "avg_xentropy":
            output_params["N_neurons"]= self.num_output
            output_params["trial_steps"]= self.trial_steps
            output_params["N_class"]= self.N_class

        self.output_init_vars= {"V": p["V_RESET"],
                                "lambda_V": 0.0,
                                "lambda_I": 0.0,
                                "trial": 0,
        }
        if p["LOSS_TYPE"] == "max":
            self.output_init_vars["max_V"]= p["V_RESET"]
            self.output_init_vars["new_max_V"]= p["V_RESET"]
            self.output_init_vars["max_t"]= 0.0
            self.output_init_vars["new_max_t"]= 0.0
            self.output_init_vars["rev_t"]= 0.0
            self.output_init_vars["expsum"]= 1.0
            self.output_init_vars["exp_V"]= 1.0     

        if p["LOSS_TYPE"] == "sum":
            self.output_init_vars["sum_V"]= 0.0
            self.output_init_vars["new_sum_V"]= 0.0
            self.output_init_vars["rev_t"]= 0.0
            self.output_init_vars["expsum"]= 1.0
            self.output_init_vars["exp_V"]= 1.0     

        if p["LOSS_TYPE"] == "avg_xentropy":
            self.output_init_vars["sum_V"]= 0.0
            self.output_init_vars["rp_V"]= 0
            self.output_init_vars["wp_V"]= 0
            self.output_init_vars["loss"]= 0
            
        # ----------------------------------------------------------------------------
        # Synapse initialisation
        # ----------------------------------------------------------------------------
        self.in_to_hid_init_vars= {"dw": 0}
        self.in_to_hid_init_vars["w"]= genn_model.init_var("Normal", {"mean": p["INPUT_HIDDEN_MEAN"], "sd": p["INPUT_HIDDEN_STD"]})

        self.hid_to_out_init_vars= {"dw": 0}
        self.hid_to_out_init_vars["w"]= genn_model.init_var("Normal", {"mean": p["HIDDEN_OUTPUT_MEAN"], "sd": p["HIDDEN_OUTPUT_STD"]})

        if p["RECURRENT"]:
            self.hid_to_hid_init_vars= {"dw": 0}
            self.hid_to_hid_init_vars["w"]= genn_model.init_var("Normal", {"mean": p["HIDDEN_HIDDEN_MEAN"], "sd": p["HIDDEN_HIDDEN_STD"]})
        # ----------------------------------------------------------------------------
        # Optimiser initialisation
        # ----------------------------------------------------------------------------
        adam_params = {"beta1": p["ADAM_BETA1"], "beta2": p["ADAM_BETA2"], "epsilon": p["ADAM_EPS"], "tau_syn": p["TAU_SYN"], "N_batch": p["N_BATCH"]}
        self.adam_init_vars = {"m": 0.0, "v": 0.0}

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

        # Add neuron populations
        self.input = self.model.add_neuron_population("input", self.num_input, EVP_SSA_MNIST_SHUFFLE, 
                                                      input_params, self.input_init_vars)
        self.input.set_extra_global_param("t_k",-1e5*np.ones(p["N_BATCH"]*self.num_input*p["N_MAX_SPIKE"],dtype=np.float32))
        self.input.set_extra_global_param("spikeTimes", np.zeros(200000000,dtype=np.float32)) # reserve enough space for any set of input spikes that is likely

        if p["REG_TYPE"] == "simple":
            hidden_params["N_batch"]= p["N_BATCH"]
            hidden_params["lbd_upper"]= p["LBD_UPPER"]
            hidden_params["lbd_lower"]= p["LBD_LOWER"]
            hidden_params["nu_upper"]= p["NU_UPPER"]
            self.hidden_init_vars["sNSum"]= 0.0
            self.hidden_init_vars["new_sNSum"]= 0.0
            self.hidden= self.model.add_neuron_population("hidden", p["NUM_HIDDEN"], EVP_LIF_reg, hidden_params, self.hidden_init_vars) 
        if p["REG_TYPE"] == "Thomas1":
            hidden_params["N_batch"]= p["N_BATCH"]
            hidden_params["lbd_lower"]= p["LBD_LOWER"]
            hidden_params["nu_lower"]= p["NU_LOWER"]
            hidden_params["lbd_upper"]= p["LBD_UPPER"]
            hidden_params["nu_upper"]= p["NU_UPPER"]
            hidden_params["rho_upper"]= p["RHO_UPPER"]
            hidden_params["glb_upper"]= p["GLB_UPPER"]
            hidden_params["N_batch"]= p["N_BATCH"]
            self.hidden_init_vars["sNSum"]= 0.0
            self.hidden_init_vars["new_sNSum"]= 0.0
            self.hidden= self.model.add_neuron_population("hidden", p["NUM_HIDDEN"], EVP_LIF_reg_Thomas1, hidden_params, self.hidden_init_vars) 
        if p["REG_TYPE"] == "none":
            self.hidden= self.model.add_neuron_population("hidden", p["NUM_HIDDEN"], EVP_LIF, hidden_params, self.hidden_init_vars) 
        self.hidden.set_extra_global_param("t_k",-1e5*np.ones(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"],dtype=np.float32))
        self.hidden.set_extra_global_param("ImV",np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"],dtype=np.float32))
        if p["REG_TYPE"] == "Thomas1":
            self.hidden.set_extra_global_param("sNSum_all", np.zeros(p["N_BATCH"]))

        if p["LOSS_TYPE"] == "max":
            self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_MNIST, output_params, self.output_init_vars)
        if p["LOSS_TYPE"] == "sum":
            self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_MNIST_sum, output_params, self.output_init_vars)
        if p["LOSS_TYPE"] == "avg_xentropy":
            self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_avg_xentropy, output_params, self.output_init_vars)
        self.output.set_extra_global_param("label", np.zeros(self.data_full_length,dtype=np.float32)) # reserve space for labels
        if p["LOSS_TYPE"] == "avg_xentropy":
            self.output.set_extra_global_param("Vbuf", np.zeros(p["N_BATCH"]*self.num_output*self.trial_steps*2,dtype=np.float32)) # reserve space for voltage buffer
        
        input_var_refs= {"rp_ImV": genn_model.create_var_ref(self.input, "rp_ImV"),
                         "wp_ImV": genn_model.create_var_ref(self.input, "wp_ImV"),
                         "back_spike": genn_model.create_var_ref(self.input, "back_spike"),
                         "fwd_start": genn_model.create_var_ref(self.input, "fwd_start"),
                         "new_fwd_start": genn_model.create_var_ref(self.input, "new_fwd_start"),
                         "rev_t": genn_model.create_var_ref(self.input, "rev_t")
        }
        self.input_reset= self.model.add_custom_update("input_reset","neuronReset", EVP_input_reset_MNIST, {"N_max_spike": p["N_MAX_SPIKE"]}, {}, input_var_refs)

        input_set_params= {"N_batch": p["N_BATCH"],
                           "num_input": self.num_input
        }
        input_var_refs= {"startSpike": genn_model.create_var_ref(self.input, "startSpike"),
                         "endSpike": genn_model.create_var_ref(self.input, "endSpike")
                         }
        self.input_set= self.model.add_custom_update("input_set", "inputUpdate", EVP_input_set_MNIST_shuffle, input_set_params, {}, input_var_refs)
        # reserving memory for the worst case of the full training set
        self.input_set.set_extra_global_param("allStartSpike", np.zeros(self.data_full_length*self.num_input,dtype= int))
        self.input_set.set_extra_global_param("allEndSpike", np.zeros(self.data_full_length*self.num_input,dtype= int))
        self.input_set.set_extra_global_param("allInputID", np.zeros(self.data_full_length,dtype= int))
        self.input_set.set_extra_global_param("trial", 0)
            
        hidden_var_refs= {"rp_ImV": genn_model.create_var_ref(self.hidden, "rp_ImV"),
                          "wp_ImV": genn_model.create_var_ref(self.hidden, "wp_ImV"),
                          "V": genn_model.create_var_ref(self.hidden, "V"),
                          "lambda_V": genn_model.create_var_ref(self.hidden, "lambda_V"),
                          "lambda_I": genn_model.create_var_ref(self.hidden, "lambda_I"),
                          "rev_t": genn_model.create_var_ref(self.hidden, "rev_t"),
                          "fwd_start": genn_model.create_var_ref(self.hidden, "fwd_start"),
                          "new_fwd_start": genn_model.create_var_ref(self.hidden, "new_fwd_start"),
                          "back_spike": genn_model.create_var_ref(self.hidden, "back_spike")
        }
        if p["REG_TYPE"] == "simple" or p["REG_TYPE"] == "Thomas1":
            hidden_var_refs["sNSum"]= genn_model.create_var_ref(self.hidden, "sNSum")
            hidden_var_refs["new_sNSum"]= genn_model.create_var_ref(self.hidden, "new_sNSum")
        if p["REG_TYPE"] == "simple":
            self.hidden_reset= self.model.add_custom_update("hidden_reset","neuronReset", EVP_neuron_reset_reg, {"V_reset": p["V_RESET"], "N_max_spike": p["N_MAX_SPIKE"], "N_neurons": p["NUM_HIDDEN"]}, {}, hidden_var_refs)
        if p["REG_TYPE"] == "Thomas1":
            self.hidden_reset= self.model.add_custom_update("hidden_reset","neuronReset", EVP_neuron_reset_reg_global, {"V_reset": p["V_RESET"], "N_max_spike": p["N_MAX_SPIKE"], "N_neurons": p["NUM_HIDDEN"]}, {}, hidden_var_refs)
            self.hidden_reset.set_extra_global_param("sNSum_all", np.zeros(p["N_BATCH"]))
        if (p["REG_TYPE"] == "simple" or p["REG_TYPE"] == "Thomas1") and p["AVG_SNSUM"]:
            var_refs= {"sNSum": genn_model.create_var_ref(self.hidden, "sNSum")}
            self.hidden_reg_reduce= self.model.add_custom_update("hidden_reg_reduce","sNSumReduce", EVP_reg_reduce, {}, {"reduced_sNSum": 0.0}, var_refs)
            var_refs= {
                "reduced_sNSum": genn_model.create_var_ref(self.hidden_reg_reduce, "reduced_sNSum"),
                "sNSum": genn_model.create_var_ref(self.hidden, "sNSum")
            }
            self.hidden_redSNSum_apply= self.model.add_custom_update("hidden_redSNSum_apply","sNSumApply", EVP_sNSum_apply, {"N_batch": p["N_BATCH"]}, {}, var_refs)
        if p["REG_TYPE"] == "none":
            self.hidden_reset= self.model.add_custom_update("hidden_reset","neuronReset", EVP_neuron_reset, {"V_reset": p["V_RESET"], "N_max_spike": p["N_MAX_SPIKE"]}, {}, hidden_var_refs)

        output_reset_params= {"V_reset": p["V_RESET"],
                              "N_class": self.N_class
        }
        if p["LOSS_TYPE"] == "avg_xentropy":
            output_reset_params["trial_steps"]= self.trial_steps
        
        output_var_refs= {"V": genn_model.create_var_ref(self.output, "V"),
                          "lambda_V": genn_model.create_var_ref(self.output, "lambda_V"),
                          "lambda_I": genn_model.create_var_ref(self.output, "lambda_I"),
                          "trial": genn_model.create_var_ref(self.output, "trial")
        }

        if p["LOSS_TYPE"] == "max":
            output_var_refs["max_V"]= genn_model.create_var_ref(self.output, "max_V")
            output_var_refs["new_max_V"]= genn_model.create_var_ref(self.output, "new_max_V")
            output_var_refs["max_t"]= genn_model.create_var_ref(self.output, "max_t")
            output_var_refs["new_max_t"]= genn_model.create_var_ref(self.output, "new_max_t")
            output_var_refs["rev_t"]= genn_model.create_var_ref(self.output, "rev_t")
            output_var_refs["expsum"]= genn_model.create_var_ref(self.output, "expsum")
            output_var_refs["exp_V"]= genn_model.create_var_ref(self.output, "exp_V")
        if p["LOSS_TYPE"] == "sum":
            output_var_refs["sum_V"]= genn_model.create_var_ref(self.output, "sum_V")
            output_var_refs["new_sum_V"]= genn_model.create_var_ref(self.output, "new_sum_V")
            output_var_refs["rev_t"]= genn_model.create_var_ref(self.output, "rev_t")
            output_var_refs["expsum"]= genn_model.create_var_ref(self.output, "expsum")
            output_var_refs["exp_V"]= genn_model.create_var_ref(self.output, "exp_V")
        if p["LOSS_TYPE"] == "avg_xentropy":
            output_var_refs["sum_V"]= genn_model.create_var_ref(self.output, "sum_V")
            output_var_refs["rp_V"]= genn_model.create_var_ref(self.output, "rp_V")
            output_var_refs["wp_V"]= genn_model.create_var_ref(self.output, "wp_V")
            output_var_refs["loss"]= genn_model.create_var_ref(self.output, "loss")
            
        if p["LOSS_TYPE"] == "max":
            self.output_reset= self.model.add_custom_update("output_reset","neuronReset", EVP_neuron_reset_output_SHD, output_reset_params, {}, output_var_refs)
        if p["LOSS_TYPE"] == "sum":
            self.output_reset= self.model.add_custom_update("output_reset","neuronReset", EVP_neuron_reset_output_SHD_sum, output_reset_params, {}, output_var_refs)
        if p["LOSS_TYPE"] == "avg_xentropy":
            self.output_reset= self.model.add_custom_update("output_reset","neuronReset", EVP_neuron_reset_output_avg_xentropy, output_reset_params, {}, output_var_refs)

        # synapse populations
        self.in_to_hid= self.model.add_synapse_population("in_to_hid", "DENSE_INDIVIDUALG", NO_DELAY, self.input, self.hidden, EVP_input_synapse,
                                                          {}, self.in_to_hid_init_vars, {}, {}, my_Exp_Curr, {"tau": p["TAU_SYN"]}, {})
        
        self.hid_to_out= self.model.add_synapse_population("hid_to_out", "DENSE_INDIVIDUALG", NO_DELAY, self.hidden, self.output, EVP_synapse,
                                                           {}, self.hid_to_out_init_vars, {}, {}, my_Exp_Curr, {"tau": p["TAU_SYN"]}, {})
        
        if p["RECURRENT"]:
            self.hid_to_hid= self.model.add_synapse_population("hid_to_hid", "DENSE_INDIVIDUALG", NO_DELAY, self.hidden, self.hidden, EVP_synapse,
                                                               {}, self.hid_to_hid_init_vars, {}, {}, my_Exp_Curr, {"tau": p["TAU_SYN"]}, {})

        self.optimisers= []
        var_refs = {"dw": genn_model.create_wu_var_ref(self.in_to_hid, "dw")}
        self.in_to_hid_reduce= self.model.add_custom_update("in_to_hid_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
        var_refs = {"gradient": genn_model.create_wu_var_ref(self.in_to_hid_reduce, "reduced_dw"),
                    "variable": genn_model.create_wu_var_ref(self.in_to_hid, "w")}
        self.in_to_hid_learn= self.model.add_custom_update("in_to_hid_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)
        self.optimisers.append(self.in_to_hid_learn)
        
        var_refs = {"dw": genn_model.create_wu_var_ref(self.hid_to_out, "dw")}
        self.hid_to_out_reduce= self.model.add_custom_update("hid_to_out_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
        var_refs = {"gradient": genn_model.create_wu_var_ref(self.hid_to_out_reduce, "reduced_dw"),
                    "variable": genn_model.create_wu_var_ref(self.hid_to_out, "w")}
        self.hid_to_out_learn= self.model.add_custom_update("hid_to_out_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)
        self.hid_to_out.pre_target_var= "revIsyn"
        self.optimisers.append(self.hid_to_out_learn)

        if p["RECURRENT"]:
            var_refs = {"dw": genn_model.create_wu_var_ref(self.hid_to_hid, "dw")}
            self.hid_to_hid_reduce= self.model.add_custom_update("hid_to_hid_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
            var_refs = {"gradient": genn_model.create_wu_var_ref(self.hid_to_hid_reduce, "reduced_dw"),
                        "variable": genn_model.create_wu_var_ref(self.hid_to_hid, "w")}
            self.hid_to_hid_learn= self.model.add_custom_update("hid_to_hid_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)
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
            
    def run_model(self, number_epochs, p, shuffle, X_t_orig= None, labels= None, X_t_eval= None, labels_eval= None, resfile= None):
        if p["LOAD_LAST"]:
            self.in_to_hid.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_input_hidden_last.npy"))
            self.in_to_hid.push_var_to_device("w")
            self.hid_to_out.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_output_last.npy"))
            self.hid_to_out.push_var_to_device("w")
            if p["RECURRENT"]:
                self.hid_to_hid.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_hidden_last.npy"))
                self.hid_to_hid.push_var_to_device("w")
        else:
            # zero the weights of synapses to "padding output neurons" - this should remove them from influencing the backward pass
            mask= np.zeros((p["NUM_HIDDEN"],self.num_output))
            mask[:,self.N_class:]= 1
            mask= np.array(mask, dtype= bool).flatten()
            self.hid_to_out.pull_var_from_device("w")
            self.hid_to_out.vars["w"].view[mask]= 0
            self.hid_to_out.push_var_to_device("w")
            print("connections zeroed")
        # set up run
        N_trial= 0
        if X_t_orig is not None:
            assert(labels is not None)
            N_train= len(X_t_orig) // p["N_BATCH"]
            N_trial+= N_train
        else:
            N_train= 0
        if X_t_eval is not None:
            assert(labels_eval is not None)
            N_eval= len(X_t_eval) // p["N_BATCH"]
            N_trial+= N_eval
        else:
            N_eval= 0
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
        X, Y, input_start, input_end= self.generate_input_spiketimes_shuffle_fast(p, X_t_orig, labels, X_t_eval, labels_eval)
        self.input.extra_global_params["spikeTimes"].view[:len(X)]= X
        self.input.push_extra_global_param_to_device("spikeTimes")
        self.input_set.extra_global_params["allStartSpike"].view[:len(input_start)]= input_start
        self.input_set.push_extra_global_param_to_device("allStartSpike")
        self.input_set.extra_global_params["allEndSpike"].view[:len(input_end)]= input_end
        self.input_set.push_extra_global_param_to_device("allEndSpike")
        if labels is not None:
            input_id= np.arange(labels.shape[0])
        else:
            input_id= []
        all_input_id= np.arange(Y.shape[0])
        self.input_set.extra_global_params["allInputID"].view[:len(all_input_id)]= all_input_id
        self.input_set.push_extra_global_param_to_device("allInputID")
        for epoch in range(number_epochs):

            # if we are doing augmentation, the entire spike time array needs to be set up anew.
            if N_train > 0 and len(p["AUGMENTATION"]) > 0:
                lX= copy.deepcopy(X_t_orig)
                for aug in p["AUGMENTATION"]:
                    if aug == "random_shift":
                        lX= random_shift(lX,self.datarng, p["AUGMENTATION"][aug])
                    if aug == "random_dilate":
                        lX= random_dilate(lX,self.datarng, p["AUGMENTATION"][aug][0], p["AUGMENTATION"][aug][1])
                    if aug == "ID_jitter":
                        lX= ID_jitter(lX,self.datarng, p["AUGMENTATION"][aug])
                X, Y, input_start, input_end= self.generate_input_spiketimes_shuffle_fast(p, lX, labels, X_t_eval, labels_eval)
                self.input.extra_global_params["spikeTimes"].view[:len(X)]= X
                self.input.push_extra_global_param_to_device("spikeTimes")
                self.input_set.extra_global_params["allStartSpike"].view[:len(input_start)]= input_start
                self.input_set.push_extra_global_param_to_device("allStartSpike")
                self.input_set.extra_global_params["allEndSpike"].view[:len(input_end)]= input_end
                self.input_set.push_extra_global_param_to_device("allEndSpike")
            
            if N_train > 0 and shuffle:
                self.datarng.shuffle(input_id)
                all_input_id[:len(input_id)]= input_id
                Y[:len(input_id)]= labels[input_id]
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
            if p["DATASET"] == "SHD":
                self.hidden.extra_global_params["pDrop"].view[:]= p["PDROP_HIDDEN"] 
            for var, val in self.output_init_vars.items():
                self.output.vars[var].view[:]= val
            self.output.push_state_to_device()
            self.model.custom_update("EVPReduce")  # this zeros dw (so as to ignore eval gradients from last epoch!
            if p["DEBUG_HIDDEN_N"]:
                all_hidden_n= []
                all_sNSum= []
            if p["DATASET"] == "SHD":
                if p["REWIRE_SILENT"]:
                    rewire_sNSum= []
            for trial in range(N_trial):
                trial_end= (trial+1)*p["TRIAL_MS"]
                # assign the input spike train and corresponding labels
                if trial < N_train:
                    phase= "train"
                else:
                    phase= "eval"
                    self.input.extra_global_params["pDrop"].view[:]= 0.0
                    if p["DATASET"] == "SHD":
                        self.hidden.extra_global_params["pDrop"].view[:]= 0.0
                self.input_set.extra_global_params["trial"].view[:]= trial
                self.model.custom_update("inputUpdate")
                self.input.extra_global_params["t_offset"].view[:]= self.model.t
                    
                int_t= 0
                if p["DEBUG_HIDDEN_N"]:
                    if p["REG_TYPE"] != "Thomas1":
                        spike_N_hidden= np.zeros(p["N_BATCH"])
                while (self.model.t < trial_end-1e-1*p["DT_MS"]):
                    self.model.step_time()
                    int_t += 1
                    # DEBUG of middle layer activity
                    if p["DEBUG_HIDDEN_N"]:
                        if int_t%p["SPK_REC_STEPS"] == 0:
                            if p["REG_TYPE"] != "Thomas1":
                                self.model.pull_recording_buffers_from_device()
                                x= self.model.neuron_populations["hidden"].spike_recording_data
                                for btch in range(p["N_BATCH"]):
                                    spike_N_hidden[btch]+= len(x[btch][0])
                    if ((epoch,trial) in p["REC_SPIKES_EPOCH_TRIAL"]) and (len(p["REC_SPIKES"]) > 0):
                        if int_t%p["SPK_REC_STEPS"] == 0:
                            self.model.pull_recording_buffers_from_device()
                            for pop in p["REC_SPIKES"]:
                                the_pop= self.model.neuron_populations[pop]
                                x= the_pop.spike_recording_data
                                if p["N_BATCH"] > 1:
                                    for i in range(p["N_BATCH"]):
                                        spike_t[pop].append(x[i][0]+(epoch*N_trial*p["N_BATCH"]+trial*p["N_BATCH"]+i-trial)*p["TRIAL_MS"]) # subtracting trial to compensate the progression of model.t by p["TRIAL_MS"] each trial
                                        spike_ID[pop].append(x[i][1])
                                else:
                                    spike_t[pop].append(x[0]+epoch*N_trial*p["TRIAL_MS"])
                                    spike_ID[pop].append(x[1])

                    if ((epoch,trial) in p["REC_NEURONS_EPOCH_TRIAL"]):
                        for pop, var in p["REC_NEURONS"]:
                            the_pop= self.model.neuron_populations[pop]
                            the_pop.pull_var_from_device(var)
                            rec_vars_n[var+pop].append(the_pop.vars[var].view.copy())
                        rec_n_t.append(self.model.t)
                            
                    if ((epoch,trial) in p["REC_SYNAPSES_EPOCH_TRIAL"]):
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
                # do not learn after the 0th trial where lambdas are meaningless
                if (phase == "train") and trial > 0 and ((trial+1)%p["SUPER_BATCH"]) == 0:
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
                if p["LOSS_TYPE"] == "avg_xentropy": # need to copy sum_V and loss from device before neuronReset!
                    self.output.pull_var_from_device("sum_V")
                    self.output.pull_var_from_device("loss")
                self.model.custom_update("neuronReset")
                if (p["REG_TYPE"] == "simple" or p["REG_TYPE"] == "Thomas1") and p["AVG_SNSUM"]:
                    self.model.custom_update("sNSumReduce")
                    self.model.custom_update("sNSumApply")
                if p["REG_TYPE"] == "Thomas1": 
                    self.hidden_reset.pull_extra_global_param_from_device("sNSum_all")
                    #self.hidden.extra_global_params["sNSum_all"].view[:]= np.mean(self.hidden_reset.extra_global_params["sNSum_all"].view)
                    self.hidden.extra_global_params["sNSum_all"].view[:]= self.hidden_reset.extra_global_params["sNSum_all"].view[:]
                    self.hidden.push_extra_global_param_to_device("sNSum_all")
                    if p["DEBUG_HIDDEN_N"]:
                        spike_N_hidden= self.hidden_reset.extra_global_params["sNSum_all"].view[:].copy()
                # collect data for rewiring rule for silent neurons
                if p["DATASET"] == "SHD":
                    if p["REWIRE_SILENT"]:
                        self.hidden.pull_var_from_device("sNSum")
                        rewire_sNSum.append(np.sum(self.hidden.vars["sNSum"].view.copy(),axis= 0))
                # record training loss and error
                # NOTE: the neuronReset does the calculation of expsum and updates exp_V for loss types sum and max
                if p["LOSS_TYPE"] == "max" or p["LOSS_TYPE"] == "sum":
                    self.output.pull_var_from_device("exp_V")
                    #print(self.output.vars["exp_V"].view)
                    pred= np.argmax(self.output.vars["exp_V"].view[:,:self.N_class], axis=-1)
                if p["LOSS_TYPE"] == "avg_xentropy":
                    pred= np.argmax(self.output.vars["sum_V"].view[:,:self.N_class], axis=-1)
                lbl= Y[trial*p["N_BATCH"]:(trial+1)*p["N_BATCH"]]
                if ((epoch, trial) in p["REC_SPIKES_EPOCH_TRIAL"]):
                    rec_spk_lbl.append(lbl.copy())
                    rec_spk_pred.append(pred.copy())
                if ((epoch, trial) in p["REC_NEURONS_EPOCH_TRIAL"]):
                    rec_n_lbl.append(lbl.copy())
                    rec_n_pred.append(pred.copy())
                if ((epoch, trial) in p["REC_SYNAPSES_EPOCH_TRIAL"]):
                    rec_s_lbl.append(lbl.copy())
                    rec_s_pred.append(pred.copy())
                if p["DEBUG"]:
                    print(pred)
                    print(lbl)
                    print("---------------------------------------")
                if p["LOSS_TYPE"] == "max" or p["LOSS_TYPE"] == "sum":
                    self.output.pull_var_from_device("expsum")
                    losses= self.loss_func(lbl,p)   # uses self.output.vars["exp_V"].view and self.output.vars["expsum"].view
                if p["LOSS_TYPE"] == "avg_xentropy":
                    losses= self.loss_func_avg_xentropy(lbl,p)   # uses self.output.vars["loss"].view 
                if ((epoch, trial) in p["REC_NEURONS_EPOCH_TRIAL"]):
                    print(pred)
                    print(lbl)
                    print("---------------------------------------")
                    #rec_exp_V.append(self.output.vars["exp_V"].view.copy())
                    #rec_expsum.append(self.output.vars["expsum"].view.copy())
                #print(f'{np.min(self.output.vars["expsum"].view)} {np.mean(self.output.vars["expsum"].view)} {np.max(self.output.vars["expsum"].view)}')
                good[phase] += np.sum(pred == lbl)
                predict[phase].append(pred)
                the_loss[phase].append(losses)
                if p["DEBUG_HIDDEN_N"]:
                    all_hidden_n.append(spike_N_hidden)
                    self.hidden.pull_var_from_device("sNSum")
                    all_sNSum.append(np.mean(self.hidden.vars['sNSum'].view.copy(),axis= 0))
                if ((epoch, trial) in p["W_OUTPUT_EPOCH_TRIAL"]): 
                    self.in_to_hid.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_input_hidden_e{}_t{}.npy".format(epoch,trial)), self.in_to_hid.vars["w"].view.copy())
                    self.hid_to_out.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_output_e{}_t{}.npy".format(epoch,trial)), self.hid_to_out.vars["w"].view.copy())
                    if p["RECURRENT"]:
                        self.hid_to_hid.pull_var_from_device("w")
                        np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_hidden_e{}_t{}.npy".format(epoch,trial)), self.hid_to_hid.vars["w"].view.copy())
                    
            if N_train > 0:
                correct= good["train"]/(N_train*p["N_BATCH"])
            else:
                correct= 0
            if N_eval > 0:
                correct_eval= good["eval"]/(N_eval*p["N_BATCH"])
            else:
                correct_eval= 0
            n_silent= 0
            if p["REWIRE_SILENT"]:
                rewire_sNSum= np.sum(np.array(rewire_sNSum),axis= 0)
                print(rewire_sNSum.shape)
                silent= rewire_sNSum == 0
                # rewire input to hidden
                self.in_to_hid.pull_var_from_device("w")
                ith_w= self.in_to_hid.vars["w"].view[:]
                ith_w.shape= (self.num_input,p["NUM_HIDDEN"])
                n_silent= np.sum(silent)
                n_new= self.num_input*n_silent
                ith_w[:,silent]= np.reshape(rng.standard_normal(n_new)*p["INPUT_HIDDEN_STD"]+p["INPUT_HIDDEN_MEAN"], (self.num_input, n_silent))
                self.in_to_hid.push_var_to_device("w")
                ## rewire hidden to output
                #self.hid_to_out.pull_var_from_device("w")
                #hto_w= self.hid_to_out.vars["w"].view[:]
                #hto_w.shape= (p["NUM_HIDDEN"],self.num_output)
                #n_new= n_silent*self.num_output
                #hto_w[silent,:]= np.reshape(rng.standard_normal(n_new)*p["HIDDEN_OUTPUT_STD"]+p["HIDDEN_OUTPUT_MEAN"], (n_silent,self.num_output))
                #self.hid_to_out.push_var_to_device("w")
                ## rewire recurrent connections
                #if p["RECURRENT"]:
                #    self.hid_to_hid.pull_var_from_device("w")
                #    hth_w= self.hid_to_hid.vars["w"].view[:]
                #    hth_w.shape= (p["NUM_HIDDEN"],p["NUM_HIDDEN"])
                #    n_new= n_silent*p["NUM_HIDDEN"]
                #    hth_w[silent,:]= np.reshape(rng.standard_normal(n_new)*p["HIDDEN_HIDDEN_STD"]+p["HIDDEN_HIDDEN_MEAN"], (n_silent,p["NUM_HIDDEN"]))
                #    hth_w[:,silent]= np.reshape(rng.standard_normal(n_new)*p["HIDDEN_HIDDEN_STD"]+p["HIDDEN_HIDDEN_MEAN"], (p["NUM_HIDDEN"],n_silent))
                #    self.hid_to_hid.push_var_to_device("w")

            if p["DEBUG_HIDDEN_N"]:
                all_hidden_n= np.hstack(all_hidden_n)
                all_sNSum= np.hstack(all_sNSum)
                print("Hidden spikes in model per trial: {} +/- {}, min {}, max {}".format(np.mean(all_hidden_n),np.std(all_hidden_n),np.amin(all_hidden_n),np.amax(all_hidden_n)))
                print("Hidden spikes per trial per neuron across batches: {} +/- {}, min {}, max {}".format(np.mean(all_sNSum),np.std(all_sNSum),np.amin(all_sNSum),np.amax(all_sNSum)))
            print("{} Training Correct: {}, Training Loss: {}, Evaluation Correct: {}, Evaluation Loss: {}, Rewired: {}".format(epoch, correct, np.mean(the_loss["train"]), correct_eval, np.mean(the_loss["eval"]),n_silent))

            if resfile is not None:
                resfile.write("{} {} {} {} {}".format(epoch, correct, np.mean(the_loss["train"]), correct_eval, np.mean(the_loss["eval"])))
                if p["DEBUG_HIDDEN_N"]:
                    resfile.write(" {} {} {} {}".format(np.mean(all_hidden_n),np.std(all_hidden_n),np.amin(all_hidden_n),np.amax(all_hidden_n)))
                    resfile.write(" {} {} {} {} {}".format(np.mean(all_sNSum),np.std(all_sNSum),np.amin(all_sNSum),np.amax(all_sNSum),n_silent))
                resfile.write("\n")
                resfile.flush()
            predict[phase]= np.hstack(predict[phase])
            learning_rate *= p["ETA_DECAY"]
            if p["ETA_FIDDELING"]:
                if (epoch+1) % p["ETA_REDUCE_PERIOD"] == 0:
                    learning_rate *= p["ETA_REDUCE"]
                    adam_step= 1

        for pop in p["REC_SPIKES"]:
            spike_t[pop]= np.hstack(spike_t[pop])
            spike_ID[pop]= np.hstack(spike_ID[pop])

        for rec_t, rec_var, rec_list in [ (rec_n_t, rec_vars_n, p["REC_NEURONS"]), (rec_s_t, rec_vars_s, p["REC_SYNAPSES"])]:
            for pop, var in rec_list:
                rec_var[var+pop]= np.array(rec_var[var+pop])
                print(rec_var[var+pop].shape)
            rec_t= np.array(rec_t)

        #rec_exp_V= np.array(rec_exp_V)
        #rec_expsum= np.array(rec_expsum)          
        rec_spk_lbl= np.array(rec_spk_lbl)
        rec_spk_pred= np.array(rec_spk_pred)
        rec_n_lbl= np.array(rec_n_lbl)
        rec_n_pred= np.array(rec_n_pred)
        rec_s_lbl= np.array(rec_s_lbl)
        rec_s_pred= np.array(rec_s_pred)
            
        if p["WRITE_TO_DISK"]:            # Saving results
            if len(p["REC_SPIKES"]) > 0:
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_spk_lbl"), rec_spk_lbl)
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_spk_pred"), rec_spk_pred)
            for pop in p["REC_SPIKES"]:
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_"+pop+"_spike_t"), spike_t[pop])
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_"+pop+"_spike_ID"), spike_ID[pop])

            if len(p["REC_NEURONS"]) > 0:
                #np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_exp_V"), rec_exp_V)
                #np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_expsum"), rec_expsum)
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

        if p["WRITE_TO_DISK"]:            # Saving results
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
        if p["DATASET"] == "SHD" or p["DATASET"] == "enose":
            if p["EVALUATION"] == "random":
                X_train, Y_train, X_eval, Y_eval= self.split_SHD_random(self.X_train_orig, self.Y_train_orig, p)
            if p["EVALUATION"] == "speaker":
                X_train, Y_train, X_eval, Y_eval= self.split_SHD_speaker(self.X_train_orig, self.Y_train_orig, self.Z_train_orig, 0, p)
            return self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_t_orig= X_train, labels= Y_train, X_t_eval= X_eval, labels_eval= Y_eval, resfile= resfile)
        if p["DATASET"] == "MNIST":
            return self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_t_orig= self.X_train_orig, labels= self.Y_train_orig, X_t_eval= self.X_val_orig, labels_eval= self.Y_val_orig, resfile= resfile)
        if p["DATASET"] == "enose":
            X_train= self.X_train_orig
            Y_train= self.Y_train_orig
            X_eval= []
            Y_eval= []
            return self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_t_orig= X_train, labels= Y_train, X_t_eval= X_eval, labels_eval= Y_eval, resfile= resfile)
        
    def cross_validate_SHD(self, p):
        resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "a")
        speakers= set(self.Z_train_orig)
        all_res= []
        times= []
        for i in speakers:
            start_t= perf_counter()
            self.define_model(p, p["SHUFFLE"])
            if p["BUILD"]:
                self.model.build()
            self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
            X_train, Y_train, X_eval, Y_eval= self.split_SHD_speaker(self.X_train_orig, self.Y_train_orig, self.Z_train_orig, i, p)
            res= self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_t_orig= X_train, labels= Y_train, X_t_eval= X_eval, labels_eval= Y_eval, resfile= resfile)
            all_res.append([ res[4], res[5] ])
            times.append(perf_counter()-start_t)
        return all_res, times

    """
    def crossvalidate_enose(self, p):
        self.define_model(p, p["SHUFFLE"])
        if p["BUILD"]:
            self.model.build()
        resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "a")
        file_train = p["FILE_TRAIN"]   # -> add in p
        data_loader = EnoseDataLoader(file_train)
        folds = data_loader.get_train_val(kfold_cv=p["N_FOLDS"]) # -> add in p
        
        all_res = []
        for fold in folds:
            X_train, y_train, X_val, y_val = fold
            X_train, y_train = data_loader.format_genn(X_train, y_train)
            X_val, y_val = data_loader.format_genn(X_val, y_val)
            
            res = self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_t_orig= X_train, labels= y_train, X_t_eval= X_val, labels_eval= y_val, resfile= resfile)
            all_res.append([ res[4], res[5] ])
        return all_res
    """
    
    def test(self, p):
        self.define_model(p, False)
        if p["BUILD"]:
            self.model.build()
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        return self.run_model(1, p, False, X_t_eval= self.X_test_orig, labels_eval= self.Y_test_orig)
        
