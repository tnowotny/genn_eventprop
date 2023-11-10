import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from utils import random_shift, random_dilate, ID_jitter, blend, blend_dataset
import tonic
from models import *
import os
import urllib.request
import gzip, shutil
from tensorflow.keras.utils import get_file
import tables
import copy
from time import perf_counter
from dataclasses import dataclass
from typing import Tuple
import sys
from tqdm import tqdm

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
p["TRAIN_DATA_SEED"]= 123
p["TEST_DATA_SEED"]= 456
p["MODEL_SEED"]= None

# Experiment parameters
p["TRIAL_MS"]= 1400.0
p["N_MAX_SPIKE"]= 400    # make buffers for maximally 400 spikes (200 in a 30 ms trial) - should be safe
p["N_BATCH"]= 32
p["N_TRAIN"]= 7644 # together with N_VALIDATE= 512 this is all 8156 samples
p["N_VALIDATE"]= 512
p["N_EPOCH"]= 100
p["SHUFFLE"]= True
#p["N_TEST"]= 2264 - just for reference, simulator will derive from the data

# Network structure
p["NUM_HIDDEN"] = 256
p["RECURRENT"] = False

# Model parameters
p["TAU_SYN"] = 5.0
p["TAU_MEM"] = 20.0
p["TAU_MEM_OUTPUT"] = 20.0
p["V_THRESH"] = 1.0
p["V_RESET"] = 0.0
p["PDROP_INPUT"] = 0.1
p["PDROP_HIDDEN"] = 0.0

# Regularisation related parameters
p["REG_TYPE"]= "none"
p["LBD_UPPER"]= 2e-9
p["LBD_LOWER"]= 2e-9
p["NU_UPPER"]= 14
p["NU_LOWER"]= 5
p["RHO_UPPER"]= 10000.0
p["GLB_UPPER"]= 1e-8
p["REWIRE_SILENT"]= False
p["REWIRE_LIFT"]= 0.0

# ALIF related parameters
p["HIDDEN_NEURON_TYPE"]= "LIF"
p["TAU_B"] = 100.0
p["B_INCR"]= 0.1
p["B_INIT"]= 0.0

# synapse related parameters
p["INPUT_HIDDEN_MEAN"]= 0.02
p["INPUT_HIDDEN_STD"]= 0.01
p["HIDDEN_OUTPUT_MEAN"]= 0.0
p["HIDDEN_OUTPUT_STD"]= 0.3
p["HIDDEN_HIDDEN_MEAN"]= 0.0   # only used when recurrent
p["HIDDEN_HIDDEN_STD"]= 0.02   # only used when recurrent
p["HIDDEN_HIDDENFWD_MEAN"]= 0.02 # only used when > 1 hidden layer
p["HIDDEN_HIDDENFWD_STD"]= 0.01 # only used when > 1 hidden layer

# Learning parameters
p["ETA"]= 1e-3
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

# possible loss types: "first_spike", "first_spike_exp", "max",
# "sum", "sum_weigh_linear", "sum_weigh_exp", "sum_weigh_sigmoid", "sum_weigh_input",
# "avg_xentropy"
p["LOSS_TYPE"]= "sum_weigh_exp"

# possible evaluation types: "random", "speaker"
p["EVALUATION"]= "random"
p["CUDA_VISIBLE_DEVICES"]= False
p["AVG_SNSUM"]= False
p["REDUCED_CLASSES"]= None
p["AUGMENTATION"]= {}
p["DOWNLOAD_SHD"]= False
p["COLLECT_CONFUSION"]= False
p["REC_PREDICTIONS"]= False
# "first_spike" loss function variables
p["TAU_0"]= 0.5
p["TAU_1"]= 6.4
p["ALPHA"]= 3e-3

# for input-weighted sum losses
p["TAU_ACCUMULATOR"]= 20.0

# Gaussian noise on hidden neurons' membrane potential
p["HIDDEN_NOISE"]= 0.0

p["SPEAKER_LEFT"]= 0

# rescaling factor for the time, 1.0 means no rescaling
p["RESCALE_T"]= 1.0
# rescaling factor for the channels, 1.0 means no rescaling
p["RESCALE_X"]= 1.0

# whether to train neuron timescales
p["TRAIN_TAUM"]= False
p["MIN_TAU_M"]= 1.0
p["N_HID_LAYER"]= 1

# learning rate schedule depending on exponential moving average of performance
p["EMA_ALPHA1"]= 0.8
p["EMA_ALPHA2"]= 0.95
p["ETA_FAC"]= 0.5
p["MIN_EPOCH_ETA_FIXED"]= 300

p["TAUM_OUTPUT_EPOCH_TRIAL"]= []
p["AUGMENTATION"]["NORMALISE_SPIKE_NUMBER"]= False
p["BALANCE_TRAIN_CLASSES"]= False
p["BALANCE_EVAL_CLASSES"]= False

p["DATA_SET"]= "SHD"

p["DATA_BUFFER_NAME"]= "./data/SSC/mySSC"

p["N_INPUT_DELAY"]= 0
p["INPUT_DELAY"]= 50.0 # in ms


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

rng= np.random.default_rng()

@dataclass
class EventsToGrid:
    sensor_size: Tuple[int, int, int]
    dt: float

    def __call__(self, events):
        # Tuple of possible axis names
        axes = ("x", "y", "p")

        # Build bin and sample data structures for histogramdd
        bins = []
        sample = []
        for s, a in zip(self.sensor_size, axes):
            if a in events.dtype.names:
                bins.append(np.linspace(0, s, s + 1))
                sample.append(events[a])

        # Add time bins
        bins.append(np.arange(0.0, np.amax(events["t"]) + self.dt, self.dt))
        sample.append(events["t"])

        # Build histogram
        event_hist,_ = np.histogramdd(tuple(sample), tuple(bins))
        new_events = np.where(event_hist > 0)

        # Copy x, y, p data into new structured array
        grid_events = np.empty(len(new_events[0]), dtype=events.dtype)
        for i, a in enumerate(axes):
            if a in grid_events.dtype.names:
                grid_events[a] = new_events[i]

        # Add t, scaling by dt
        grid_events["t"] = new_events[-1] * self.dt
        return grid_events

def rescale(x, t, p):
    new_x= np.array(x*p["RESCALE_X"])
    new_x= np.floor(new_x).astype(int)
    new_t= np.array(t*p["RESCALE_T"]) 
    which= new_t < p["TRIAL_MS"]
    new_x= new_x[which]
    new_t= new_t[which]
    new_t= np.floor(new_t/p["DT_MS"]).astype(int)
    fmatrix= np.zeros((int(700*p["RESCALE_X"]),int(p["TRIAL_MS"]/p["DT_MS"])))
    fmatrix[new_x, new_t]= 1
    idx= np.where(fmatrix == 1)
    sample= {"x": idx[0], "t": idx[1]*p["DT_MS"]}
    return sample

def update_adam(learning_rate, adam_step, optimiser_custom_updates):
    first_moment_scale = 1.0 / (1.0 - (p["ADAM_BETA1"] ** adam_step))
    second_moment_scale = 1.0 / (1.0 - (p["ADAM_BETA2"] ** adam_step))

    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["alpha"].view[:] = learning_rate
        o.extra_global_params["firstMomentScale"].view[:] = first_moment_scale
        o.extra_global_params["secondMomentScale"].view[:] = second_moment_scale

class SHD_model:

    def __init__(self, p, manual_GPU= None):
        self.GPU= manual_GPU
        if p["TRAIN_DATA_SEED"] is not None:
            self.datarng= np.random.default_rng(p["TRAIN_DATA_SEED"])
        else:
            self.datarng= np.random.default_rng()

        if p["TEST_DATA_SEED"] is not None:
            self.tdatarng= np.random.default_rng(p["TEST_DATA_SEED"])
        else:
            self.tdatarng= np.random.default_rng()        
            
        print("loading data ...")
        #self.load_data_SHD_Zenke(p)
        if p["DATA_SET"] == "SHD":
            self.load_data_SHD(p)
        if p["DATA_SET"] == "SSC":
            self.load_data_SSC(p)
        print("loading data complete ...")
        
        if p["REDUCED_CLASSES"] is not None and len(p["REDUCED_CLASSES"]) > 0:
            self.X_train_orig, self.Y_train_orig, self.Z_train_orig= self.reduce_classes(self.X_train_orig, self.Y_train_orig, self.Z_train_orig, p["REDUCED_CLASSES"])
            self.X_test_orig, self.Y_test_orig, self.Z_test_orig= self.reduce_classes(self.X_test_orig, self.Y_test_orig, self.Z_test_orig, p["REDUCED_CLASSES"])


    def plot_examples(self,spkrs,digit,nsample,phase):
        if phase == "train":
            X= self.X_train_orig
            Y= self.Y_train_orig
            Z= self.Z_train_orig
        else:
            X= self.X_test_orig
            Y= self.Y_test_orig
            Z= self.Z_test_orig

        if p["AUGMENTATION"]["NORMALISE_SPIKE_NUMBER"]:
            X= self.normalise_spike_number(X)
        ydim= len(spkrs)
        xdim= nsample
        if ydim > 0:
            fig, ax= plt.subplots(ydim,xdim,sharex= True, sharey= True)
            for y in range(ydim):
                td= X[np.logical_and(Z == spkrs[y], Y == digit)][:nsample]
                for x in range(len(td)):
                    ax[y,x].scatter(td[x]["t"],td[x]["x"],s=0.1)
        else: # no speaker info
            fig, ax= plt.subplots(5,xdim,sharex= True, sharey= True)
            for y in range(5):
                td= X[Y == digit][:nsample]
                for x in range(len(td)):
                    ax[y,x].scatter(td[x]["t"],td[x]["x"],s=0.1)
 
        plt.show()

    def plot_example_means(self,spkrs,digits,nsample,phase,p):
        if phase == "train":
            X= self.X_train_orig
            Y= self.Y_train_orig
            Z= self.Z_train_orig
        else:
            X= self.X_test_orig
            Y= self.Y_test_orig
            Z= self.Z_test_orig

        if p["AUGMENTATION"]["NORMALISE_SPIKE_NUMBER"]:
            X= self.normalise_spike_number(X)
        ydim= len(spkrs)
        xdim= len(digits)
        fig, ax= plt.subplots(ydim,xdim,sharex= True, sharey= True)
        mat= np.zeros
        for y in range(ydim):
            for x in range(xdim):
                td= X[np.logical_and(Z == spkrs[y], Y == digits[x])][:nsample]
                mat= np.zeros((self.num_input,int(p["TRIAL_MS"]/p["DT_MS"])))
                for spk in td:
                    mat[spk["x"].astype(int),spk["t"].astype(int)]+= 1.0
                im= ax[y,x].imshow(mat, vmin= 0, vmax= 20, cmap="jet")
                ratio = 0.5
                xleft, xright = ax[y,x].get_xlim()
                ybottom, ytop = ax[y,x].get_ylim()
                ax[y,x].set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
                #plt.colorbar(im, ax= ax[y,x])
        plt.show()

    def calc_example_stats(self,spkrs,digits,phase,p):
        if phase == "train":
            X= self.X_train_orig
            Y= self.Y_train_orig
            Z= self.Z_train_orig
        else:
            X= self.X_test_orig
            Y= self.Y_test_orig
            Z= self.Z_test_orig
            
        if p["AUGMENTATION"]["NORMALISE_SPIKE_NUMBER"]:
            X= self.normalise_spike_number(X)
        mn= np.zeros((len(spkrs),len(digits)))
        n= np.zeros(mn.shape)
        sq= np.zeros(mn.shape)
        sd= np.zeros(mn.shape)
        sidx= { s: i for i,s in enumerate(spkrs) }
        didx= { d: i for i,d in enumerate(digits) }
        for i in range(len(X)):
            sn= len(X[i]["t"])
            mn[sidx[Z[i]],didx[Y[i]]]+= sn
            n[sidx[Z[i]],didx[Y[i]]]+= 1
            sq[sidx[Z[i]],didx[Y[i]]]+= sn*sn
        for i in range(mn.shape[0]):
            for j in range(mn.shape[1]):
                mn[i,j]/= n[i,j]
                sd[i,j]= np.sqrt(sq[i,j]/n[i,j]-mn[i,j]*mn[i,j])
        return (mn, sd, n)
            

        
    def plot_blend_examples(self,spkrs,digits,probs,phase,p):
        if phase == "train":
            X= self.X_train_orig
            Y= self.Y_train_orig
            Z= self.Z_train_orig
        else:
            X= self.X_test_orig
            Y= self.Y_test_orig
            Z= self.Z_test_orig
        if p["AUGMENTATION"]["NORMALISE_SPIKE_NUMBER"]:
            X= self.normalise_spike_number(X)
        ydim= len(spkrs)
        xdim= len(digits)
        fig, ax= plt.subplots(ydim,xdim,sharex= True, sharey= True)
        for y in range(ydim):
            for x in range(xdim):
                td= X[np.logical_and(Z == spkrs[y], Y == digits[x])][:len(probs)]
                ntd= blend(td,probs,self.datarng,p)
                ax[y,x].scatter(ntd["t"],ntd["x"],s=0.1)
        plt.show()
        
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
            print(softmaxVal_correct)
            print(softmaxVal[np.where(softmaxVal_correct == 0),:])
            softMaxVal_correct[softmaxVal_correct == 0]+= 2e-45 # make sure all exp_V are > 0
        loss= -np.sum(np.log(SoftmaxVal_correct))/N_batch
        return loss

    def loss_func_avg_xentropy(self, N_batch):
        loss= self.output.vars["loss"].view[:N_batch,:self.N_class]
        loss= np.mean(np.sum(loss,axis= 1)) # use mean to achieve 1/N_batch here
        return loss

    def load_data_SHD(self, p):
        if p["TRAIN_DATA_SEED"] is not None:
            self.datarng= np.random.default_rng(p["TRAIN_DATA_SEED"])
        else:
            self.datarng= np.random.default_rng()        
        if p["TEST_DATA_SEED"] is not None:
            self.tdatarng= np.random.default_rng(p["TEST_DATA_SEED"])
        else:
            self.tdatarng= np.random.default_rng()        
        dataset = tonic.datasets.SHD(save_to='./data', train=True, transform=tonic.transforms.Compose([tonic.transforms.CropTime(max=1000.0 * 1000.0), EventsToGrid(tonic.datasets.SHD.sensor_size, p["DT_MS"] * 1000.0)]))
        sensor_size = dataset.sensor_size
        self.data_max_length= max(len(dataset),p["N_TRAIN"])+2*p["N_BATCH"]
        self.N_class= len(dataset.classes)
        self.num_input= int(np.product(sensor_size))
        self.num_output= self.N_class
        self.Y_train_orig= np.empty(len(dataset), dtype= int)
        self.X_train_orig= []
        for i in range(len(dataset)):
            events, label = dataset[i]
            self.Y_train_orig[i]= label
            if p["RESCALE_X"] != 1.0 or p["RESCALE_T"] != 1.0:
                sample= rescale(events["x"], events["t"]/1000.0, p)
            else:
                sample= {"x": events["x"], "t": events["t"]/1000.0}
            self.X_train_orig.append(sample)
        self.X_train_orig= np.array(self.X_train_orig)
        self.Z_train_orig= dataset.speaker
        dataset = tonic.datasets.SHD(save_to='./data', train=False, transform=tonic.transforms.Compose([tonic.transforms.CropTime(max=1000.0 * 1000.0), EventsToGrid(tonic.datasets.SHD.sensor_size, p["DT_MS"] * 1000.0)]))
        self.data_max_length+= len(dataset)
        self.Y_test_orig= np.empty(len(dataset), dtype= int)
        self.X_test_orig= []
        for i in range(len(dataset)):
            events, label = dataset[i]
            self.Y_test_orig[i]= label
            if p["RESCALE_X"] != 1.0 or p["RESCALE_T"] != 1.0:
                sample= rescale(events["x"], events["t"]/1000.0, p)
            else:
                sample= {"x": events["x"], "t": events["t"]/1000.0}
            self.X_test_orig.append(sample)
        self.X_test_orig= np.array(self.X_test_orig)
        self.Z_test_orig= dataset.speaker
    
    def load_data_SSC(self, p):
        if p["TRAIN_DATA_SEED"] is not None:
            self.datarng= np.random.default_rng(p["TRAIN_DATA_SEED"])
        else:
            self.datarng= np.random.default_rng()        
        if p["TEST_DATA_SEED"] is not None:
            self.tdatarng= np.random.default_rng(p["TEST_DATA_SEED"])
        else:
            self.tdatarng= np.random.default_rng()        
           
        dataset = tonic.datasets.SSC(save_to='./data', split="train", transform=tonic.transforms.Compose([tonic.transforms.CropTime(max=1000.0 * 1000.0), EventsToGrid(tonic.datasets.SSC.sensor_size, p["DT_MS"] * 1000.0)]))
        sensor_size = dataset.sensor_size
        self.data_max_length= len(dataset)+2*p["N_BATCH"]
        self.N_class= len(dataset.classes)
        self.num_input= int(np.product(sensor_size))
        self.num_output= self.N_class
        self.Z_train_orig= None
        self.Y_train_orig= np.empty(len(dataset), dtype= int)
        self.X_train_orig= []
        if os.path.exists(p["DATA_BUFFER_NAME"]+"_X_train_orig.npy"):
            self.X_train_orig= np.load(p["DATA_BUFFER_NAME"]+"_X_train_orig.npy",allow_pickle= True)
            self.Y_train_orig= np.load(p["DATA_BUFFER_NAME"]+"_Y_train_orig.npy",allow_pickle= True)
            print(f"data loaded from buffered file {p['DATA_BUFFER_NAME']+'_*_train_orig.npy'}")
        else:
            for i in tqdm(range(len(dataset))):
                events, label = dataset[i]
                self.Y_train_orig[i]= label
                if p["RESCALE_X"] != 1.0 or p["RESCALE_T"] != 1.0:
                    sample= rescale(events["x"], events["t"]/1000.0, p)
                else:
                    sample= {"x": events["x"], "t": events["t"]/1000.0}
                self.X_train_orig.append(sample)
            self.X_train_orig= np.array(self.X_train_orig)
            np.save(p["DATA_BUFFER_NAME"]+"_X_train_orig", self.X_train_orig, allow_pickle= True)
            np.save(p["DATA_BUFFER_NAME"]+"_Y_train_orig", self.Y_train_orig, allow_pickle= True)
            print(f"data saved to buffer file {p['DATA_BUFFER_NAME']+'_*_train_orig.npy'}")
        dataset = tonic.datasets.SSC(save_to='./data', split="valid", transform=tonic.transforms.Compose([tonic.transforms.CropTime(max=1000.0 * 1000.0), EventsToGrid(tonic.datasets.SSC.sensor_size, p["DT_MS"] * 1000.0)]))
        self.data_max_length+= len(dataset)
        self.Z_eval_orig= None
        self.Y_eval_orig= np.empty(len(dataset), dtype= int)
        self.X_eval_orig= []
        if os.path.exists(p["DATA_BUFFER_NAME"]+"_X_eval_orig.npy"):
            self.X_eval_orig= np.load(p["DATA_BUFFER_NAME"]+"_X_eval_orig.npy",allow_pickle= True)
            self.Y_eval_orig= np.load(p["DATA_BUFFER_NAME"]+"_Y_eval_orig.npy",allow_pickle= True)
            print(f"data loaded from buffered file {p['DATA_BUFFER_NAME']+'_*_eval_orig.npy'}")
        else:
            for i in tqdm(range(len(dataset))):
                events, label = dataset[i]
                self.Y_eval_orig[i]= label
                if p["RESCALE_X"] != 1.0 or p["RESCALE_T"] != 1.0:
                    sample= rescale(events["x"], events["t"]/1000.0, p)
                else:
                    sample= {"x": events["x"], "t": events["t"]/1000.0}
                self.X_eval_orig.append(sample)
            self.X_eval_orig= np.array(self.X_eval_orig)
            np.save(p["DATA_BUFFER_NAME"]+"_X_eval_orig", self.X_eval_orig, allow_pickle= True)
            np.save(p["DATA_BUFFER_NAME"]+"_Y_eval_orig", self.Y_eval_orig, allow_pickle= True)
            print(f"data saved to buffer file {p['DATA_BUFFER_NAME']+'_*_eval_orig.npy'}")
        dataset = tonic.datasets.SSC(save_to='./data', split="test", transform=tonic.transforms.Compose([tonic.transforms.CropTime(max=1000.0 * 1000.0), EventsToGrid(tonic.datasets.SSC.sensor_size, p["DT_MS"] * 1000.0)]))
        self.data_max_length+= len(dataset)
        self.Z_test_orig= None
        self.Y_test_orig= np.empty(len(dataset), dtype= int)
        self.X_test_orig= []
        if os.path.exists(p["DATA_BUFFER_NAME"]+"_X_test_orig.npy"):
            self.X_test_orig= np.load(p["DATA_BUFFER_NAME"]+"_X_test_orig.npy",allow_pickle= True)
            self.Y_test_orig= np.load(p["DATA_BUFFER_NAME"]+"_Y_test_orig.npy",allow_pickle= True)
            print(f"data loaded from buffered file {p['DATA_BUFFER_NAME']+'_*_test_orig.npy'}")
        else:
            for i in tqdm(range(len(dataset))):
                events, label = dataset[i]
                self.Y_test_orig[i]= label
                if p["RESCALE_X"] != 1.0 or p["RESCALE_T"] != 1.0:
                    sample= rescale(events["x"], events["t"]/1000.0, p)
                else:
                    sample= {"x": events["x"], "t": events["t"]/1000.0}
                self.X_test_orig.append(sample)
            self.X_test_orig= np.array(self.X_test_orig)
            np.save(p["DATA_BUFFER_NAME"]+"_X_test_orig", self.X_test_orig, allow_pickle= True)
            np.save(p["DATA_BUFFER_NAME"]+"_Y_test_orig", self.Y_test_orig, allow_pickle= True)
            print(f"data saved to buffer file {p['DATA_BUFFER_NAME']+'_*_test_orig.npy'}")

    def load_data_SHD_Zenke(self, p):
        cache_dir=os.path.expanduser("~/data")
        cache_subdir="SHD"
        print("Using cache dir: %s"%cache_dir)
        self.num_input= int(700*p["RESCALE_X"])
        self.num_output= 20
        self.data_max_length= 2*p["N_BATCH"]
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
        self.Y_train_orig= np.array(fileh.root.labels)
        self.Z_train_orig= np.array(fileh.root.extra.speaker)
        self.data_max_length+= max(len(units),p["N_TRAIN"])
        self.N_class= len(set(self.Y_train_orig))
        self.X_train_orig= []
        for i in range(len(units)):
            if p["RESCALE_X"] != 1.0 or p["RESCALE_T"] != 1.0:
                sample= rescale(units[i], times[i]*1000.0, p)
            else:
                sample= {"x": units[i], "t": times[i]*1000.0}
            self.X_train_orig.append(sample)
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
        self.data_max_length+= len(units)
        self.X_test_orig= []
        for i in range(len(units)):
            if p["RESCALE_X"] != 1.0 or p["RESCALE_T"] != 1.0:
                sample= rescale(units[i], times[i]*1000.0, p)
            else:
                sample= {"x": units[i], "t": times[i]*1000.0}
            self.X_test_orig.append(sample)
        self.X_test_orig= np.array(self.X_test_orig)            

    def normalise_spike_number(self, X, X_eval= None):
        mn= 1.0e5
        for x in X:
            if len(x["t"]) < mn:
                mn= len(x["t"])
        nx= []
        nt= []
        for i in range(len(X)):
            tprob= mn/len(X[i]["t"])
            prob= self.datarng.random(len(X[i]["t"]))
            pick= prob < tprob
            X[i]["t"]= X[i]["t"][pick]
            X[i]["x"]= X[i]["x"][pick]
        if X_eval is not None:
            for i in range(len(X_eval)):
                tprob= mn/len(X_eval[i]["t"])
                prob= self.datarng.random(len(X_eval[i]["t"]))
                pick= prob < tprob
                X_eval[i]["t"]= X_eval[i]["t"][pick]
                X_eval[i]["x"]= X_eval[i]["x"][pick]
            return X, X_eval
        return X
    
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
        train_idx= idx[:p["N_TRAIN"]]
        eval_idx= idx[p["N_TRAIN"]:p["N_TRAIN"]+p["N_VALIDATE"]]
        newX_t= X[train_idx]
        newX_e= X[eval_idx]
        newY_t= Y[train_idx]
        newY_e= Y[eval_idx]
        print(len(newX_t))
        print(len(newX_e))
        return (newX_t, newY_t, newX_e, newY_e)

    # split off one speaker to form evaluation set
    def split_SHD_speaker(self, X, Y, Z, speaker, p, shuffle= True):
        speaker= np.array(speaker)
        which= Z != speaker
        newX_t= X[which]
        newY_t= Y[which]
        newZ_t= Z[which]
        train_idx= np.arange(len(newY_t))
        if shuffle:
            self.datarng.shuffle(train_idx)
        train_idx= train_idx[:p["N_TRAIN"]]
        newX_t= newX_t[train_idx]
        newY_t= newY_t[train_idx]
        newZ_t= newZ_t[train_idx]
        which= Z == speaker
        newX_e= X[which]
        newY_e= Y[which]
        newZ_e= Z[which]
        return (newX_t, newY_t, newZ_t, newX_e, newY_e, newZ_e)

    """ 
    generate a spikeTimes array and startSpike and endSpike arrays to allow indexing into the 
    spikeTimes in a shuffled way
    """
    # ISSUE: here we are not rounding to the multiples of batch size!
    # When data loading, we are doing that for N_trial ...
    # Needs tidying up!
    def generate_input_spiketimes_shuffle_fast(self, p, Xtrain, Ytrain, Xeval, Yeval):
        Xempty= {
            "x": np.array([]),
            "t": np.array([])
        }
        Yempty= -1
        if Xtrain is None:
            X= Xeval
            Y= Yeval
            padN= p["N_BATCH"]- ((len(X)-1) % p["N_BATCH"] +1)
            X= np.append(X, [Xempty]*padN)
            Y= np.append(Y, [Yempty]*padN)
        else:
            X= Xtrain
            Y= Ytrain
            padN= p["N_BATCH"]- ((len(X)-1) % p["N_BATCH"] +1)
            X= np.append(X, [Xempty]*padN)
            Y= np.append(Y, [Yempty]*padN)
            if Xeval is not None:
                X= np.append(X, Xeval, axis= 0)    
                Y= np.append(Y, Yeval, axis= 0)
                padN= p["N_BATCH"]- ((len(X)-1) % p["N_BATCH"] +1)
                X= np.append(X, [Xempty]*padN)
                Y= np.append(Y, [Yempty]*padN)
        Y= Y.astype(int)
        # N is the number of training/testing images: always use all images given
        N= len(Y)
        all_sts= []
        all_input_end= []
        all_input_start= []
        stidx_offset= 0
        self.max_stim_time= 0.0
        for i in range(N):
            events= X[i]
            spike_event_ids = events["x"]
            i_end = np.cumsum(np.bincount(spike_event_ids.astype(int), 
                                          minlength=self.num_input))+stidx_offset    
            assert len(i_end) == self.num_input
            tx = events["t"][np.lexsort((events["t"], spike_event_ids))].astype(float)
            if len(tx) > 0:
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
        self.hidden= []
        self.hidden_reset= []
        self.hidden_reg_reduce= []
        self.hidden_redSNSum_apply= []
        self.hid_to_hidfwd= []
        self.hid_to_hidfwd_reduce= []
        self.hid_to_hidfwd_learn= []
        self.hid_to_hid= []
        self.hid_to_hid_reduce= [] 
        self.hid_to_hid_learn= []
        self.hidden_taum_reduce= []
        self.hidden_taum_learn= []
        self.trial_steps= int(round(p["TRIAL_MS"]/p["DT_MS"]))

        print("starting model definition ...")
        # ----------------------------------------------------------------------------
        # Model description
        # ----------------------------------------------------------------------------
        kwargs = {}
        if p["CUDA_VISIBLE_DEVICES"] or (self.GPU is not None):
            from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL
            kwargs["selectGPUByDeviceID"] = True
            kwargs["deviceSelectMethod"] = DeviceSelect_MANUAL
            if self.GPU is not None:
                kwargs["manualDeviceID"] = self.GPU
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
            "N_neurons": self.num_input*(p["N_INPUT_DELAY"]+1),
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
        if p["N_INPUT_DELAY"] == 0:
            print("Input neurons: EVP_SSA_MNIST_SHUFFLE")
            self.input= self.model.add_neuron_population("input", self.num_input, EVP_SSA_MNIST_SHUFFLE, input_params, self.input_init_vars)
        else:
            print("Input neurons: EVP_SSA_MNIST_SHUFFLE_DELAY")
            delay= np.zeros((p["N_INPUT_DELAY"]+1,self.num_input))
            for i in range(p["N_INPUT_DELAY"]):
                delay[i+1,:]= np.ones((1,self.num_input))*(i+1)*p["INPUT_DELAY"]
            delay= delay.flatten()
            self.input_init_vars["delay"]= delay
            self.input= self.model.add_neuron_population("input", self.num_input*(p["N_INPUT_DELAY"]+1), EVP_SSA_MNIST_SHUFFLE_DELAY, input_params, self.input_init_vars)
            
        self.input.set_extra_global_param("t_k", -1e5*np.ones(p["N_BATCH"]*self.num_input*(p["N_INPUT_DELAY"]+1)*p["N_MAX_SPIKE"], dtype=np.float32))
        # reserve enough space for any set of input spikes that is likely
        self.input.set_extra_global_param("spikeTimes", np.zeros(802000000, dtype=np.float32))

        input_reset_params= {"N_max_spike": p["N_MAX_SPIKE"]}
        input_reset_var_refs= {
            "back_spike": genn_model.create_var_ref(self.input, "back_spike"),
            "rp_ImV": genn_model.create_var_ref(self.input, "rp_ImV"),
            "wp_ImV": genn_model.create_var_ref(self.input, "wp_ImV"),
            "fwd_start": genn_model.create_var_ref(self.input, "fwd_start"),
            "new_fwd_start": genn_model.create_var_ref(self.input, "new_fwd_start"),
            "rev_t": genn_model.create_var_ref(self.input, "rev_t"),
        }
        print("Input Reset custom update: EVP_input_reset_MNIST")
        self.input_reset= self.model.add_custom_update("input_reset", "neuronReset", EVP_input_reset_MNIST, input_reset_params, {}, input_reset_var_refs)

        input_set_params= {
            "N_batch": p["N_BATCH"],
            "num_input": self.num_input,
        }
        input_set_var_refs= {
            "startSpike": genn_model.create_var_ref(self.input, "startSpike"),
            "endSpike": genn_model.create_var_ref(self.input, "endSpike"),
        }
        print("Input Set custom update: EVP_input_set_MNIST_shuffle")
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
                "tau_syn": p["TAU_SYN"],
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
            }
            for l in range(p["N_HID_LAYER"]):
                print(f"Hidden layer {l} neurons: EVP_LIF")
                self.hidden.append(self.model.add_neuron_population("hidden"+str(l), p["NUM_HIDDEN"], EVP_LIF, hidden_params, self.hidden_init_vars))
                self.hidden[l].set_extra_global_param("t_k", -1e5*np.ones(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
                self.hidden[l].set_extra_global_param("ImV", np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
            
            hidden_reset_params= {
                "V_reset": p["V_RESET"],
                "N_max_spike": p["N_MAX_SPIKE"],
            }
            for l in range(p["N_HID_LAYER"]):
                hidden_reset_var_refs= {
                    "rp_ImV": genn_model.create_var_ref(self.hidden[l], "rp_ImV"),
                    "wp_ImV": genn_model.create_var_ref(self.hidden[l], "wp_ImV"),
                    "V": genn_model.create_var_ref(self.hidden[l], "V"),
                    "lambda_V": genn_model.create_var_ref(self.hidden[l], "lambda_V"),
                    "lambda_I": genn_model.create_var_ref(self.hidden[l], "lambda_I"),
                    "rev_t": genn_model.create_var_ref(self.hidden[l], "rev_t"),
                    "fwd_start": genn_model.create_var_ref(self.hidden[l], "fwd_start"),
                    "new_fwd_start": genn_model.create_var_ref(self.hidden[l], "new_fwd_start"),
                    "back_spike": genn_model.create_var_ref(self.hidden[l], "back_spike"),
                }
                print(f"Hidden reset {l} custom update: EVP_neuron_reset")
                self.hidden_reset.append(self.model.add_custom_update("hidden_reset"+str(l), "neuronReset", EVP_neuron_reset, hidden_reset_params, {}, hidden_reset_var_refs))


        if p["REG_TYPE"] == "simple":
            hidden_params= {
                "V_thresh": p["V_THRESH"],
                "V_reset": p["V_RESET"],
                "N_neurons": p["NUM_HIDDEN"],
                "N_max_spike": p["N_MAX_SPIKE"],
                "tau_syn": p["TAU_SYN"],
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
            }
            if p["HIDDEN_NOISE"] > 0.0:
                hidden_params["tau_m"]= p["TAU_MEM"]
                for l in range(p["N_HID_LAYER"]):
                    print(f"Hidden layer {l} neurons: EVP_LIF_reg_noise")
                    self.hidden.append(self.model.add_neuron_population("hidden"+str(l), p["NUM_HIDDEN"], EVP_LIF_reg_noise, hidden_params, self.hidden_init_vars))
                    self.hidden[l].set_extra_global_param("A_noise", p["HIDDEN_NOISE"])
            else:
                if p["TRAIN_TAUM"]:
                    hidden_params["trial_t"]= p["TRIAL_MS"]
                    self.hidden_init_vars["tau_m"]= p["TAU_MEM"]
                    self.hidden_init_vars["dtaum"]= 0.0
                    self.hidden_init_vars["fImV_roff"]= int(p["TRIAL_MS"]/p["DT_MS"])
                    self.hidden_init_vars["fImV_woff"]= 0
                    for l in range(p["N_HID_LAYER"]):
                        print(f"Hidden layer {l} neurons: EVP_LIF_reg_taum")
                        self.hidden.append(self.model.add_neuron_population("hidden"+str(l), p["NUM_HIDDEN"], EVP_LIF_reg_taum, hidden_params, self.hidden_init_vars))
                        self.hidden[l].set_extra_global_param("fImV", np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*int(p["TRIAL_MS"]/p["DT_MS"])*2))
                else:
                    if p["HIDDEN_NEURON_TYPE"] == "LIF":
                        hidden_params["tau_m"]= p["TAU_MEM"]
                        for l in range(p["N_HID_LAYER"]):
                            print(f"Hidden layer {l} neurons: EVP_LIF_reg")
                            self.hidden.append(self.model.add_neuron_population("hidden"+str(l), p["NUM_HIDDEN"], EVP_LIF_reg, hidden_params, self.hidden_init_vars))
                    else:
                        hidden_params["tau_m"]= p["TAU_MEM"]
                        hidden_params["tau_B"]= p["TAU_B"]
                        hidden_params["B_incr"]= p["B_INCR"]
                        self.hidden_init_vars["B"]= p["B_INIT"]
                        self.hidden_init_vars["lambda_B"]= 0.0
                        for l in range(p["N_HID_LAYER"]):
                            print(f"Hidden layer {l} neurons: EVP_ALIF_reg")
                            self.hidden.append(self.model.add_neuron_population("hidden"+str(l), p["NUM_HIDDEN"], EVP_ALIF_reg, hidden_params, self.hidden_init_vars))
                        
            for l in range(p["N_HID_LAYER"]):
                self.hidden[l].set_extra_global_param("t_k", -1e5*np.ones(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
                self.hidden[l].set_extra_global_param("ImV", np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
            
            hidden_reset_params= {
                "V_reset": p["V_RESET"],
                "N_max_spike": p["N_MAX_SPIKE"],
                "N_neurons": p["NUM_HIDDEN"],
            }
            for l in range(p["N_HID_LAYER"]):
                hidden_reset_var_refs= {
                    "rp_ImV": genn_model.create_var_ref(self.hidden[l], "rp_ImV"),
                    "wp_ImV": genn_model.create_var_ref(self.hidden[l], "wp_ImV"),
                    "V": genn_model.create_var_ref(self.hidden[l], "V"),
                    "lambda_V": genn_model.create_var_ref(self.hidden[l], "lambda_V"),
                    "lambda_I": genn_model.create_var_ref(self.hidden[l], "lambda_I"),
                    "rev_t": genn_model.create_var_ref(self.hidden[l], "rev_t"),
                    "fwd_start": genn_model.create_var_ref(self.hidden[l], "fwd_start"),
                    "new_fwd_start": genn_model.create_var_ref(self.hidden[l], "new_fwd_start"),
                    "back_spike": genn_model.create_var_ref(self.hidden[l], "back_spike"),
                    "sNSum": genn_model.create_var_ref(self.hidden[l], "sNSum"),
                    "new_sNSum": genn_model.create_var_ref(self.hidden[l], "new_sNSum"),
                }
                if p["TRAIN_TAUM"]:
                    hidden_reset_params["trial_t"]= p["TRIAL_MS"]
                    hidden_reset_var_refs["fImV_roff"]= genn_model.create_var_ref(self.hidden[l], "fImV_roff")
                    hidden_reset_var_refs["fImV_woff"]= genn_model.create_var_ref(self.hidden[l], "fImV_woff")
                    hidden_reset_var_refs["dtaum"]= genn_model.create_var_ref(self.hidden[l], "dtaum")
                    print(f"Hidden layer {l} reset: EVP_neuron_reset_reg_taum")
                    self.hidden_reset.append(self.model.add_custom_update("hidden_reset"+str(l), "neuronReset", EVP_neuron_reset_reg_taum, hidden_reset_params, {}, hidden_reset_var_refs))
                else:
                    if p["HIDDEN_NEURON_TYPE"] == "LIF":
                        print(f"Hidden layer {l} reset: EVP_neuron_reset_reg")
                        self.hidden_reset.append(self.model.add_custom_update("hidden_reset"+str(l), "neuronReset", EVP_neuron_reset_reg, hidden_reset_params, {}, hidden_reset_var_refs))
                    else:
                        print(f"Hidden layer {l} reset: EVP_neuron_reset_reg_ALIF")
                        hidden_reset_params["B_init"]= p["B_INIT"]
                        hidden_reset_var_refs["B"]= genn_model.create_var_ref(self.hidden[l], "B")
                        hidden_reset_var_refs["lambda_B"]= genn_model.create_var_ref(self.hidden[l], "lambda_B")
                        self.hidden_reset.append(self.model.add_custom_update("hidden_reset"+str(l), "neuronReset", EVP_neuron_reset_reg_ALIF, hidden_reset_params, {}, hidden_reset_var_refs))
                        
                        
            if p["AVG_SNSUM"]:
                params= {"reduced_sNSum": 0.0}
                for l in range(p["N_HID_LAYER"]):
                    var_refs= {"sNSum": genn_model.create_var_ref(self.hidden[l], "sNSum")}
                    print(f"Hidden reg reduce {l} custom update: EVP_reg_reduce")
                    self.hidden_reg_reduce.append(self.model.add_custom_update("hidden_reg_reduce"+str(l), "sNSumReduce", EVP_reg_reduce, {}, params, var_refs))

                params= {"N_batch": p["N_BATCH"]}
                for l in range(p["N_HID_LAYER"]):
                    var_refs= {
                        "reduced_sNSum": genn_model.create_var_ref(self.hidden_reg_reduce[l], "reduced_sNSum"),
                        "sNSum": genn_model.create_var_ref(self.hidden[l], "sNSum")
                    }
                    print(f"Hidden SNSum apply {l} custom update: EVP_sNSum_apply")
                    self.hidden_redSNSum_apply.append(self.model.add_custom_update("hidden_redSNSum_apply"+str(l), "sNSumApply", EVP_sNSum_apply, params, {}, var_refs))


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
                "nu_lower": p["NU_LOWER"],
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
            for l in range(p["N_HID_LAYER"]):
                print(f"Hidden layer {l} neurons: EVP_LIF_reg_Thomas1")
                self.hidden.append(self.model.add_neuron_population("hidden"+str(l), p["NUM_HIDDEN"], EVP_LIF_reg_Thomas1, hidden_params, self.hidden_init_vars))
                self.hidden[l].set_extra_global_param("t_k", -1e5*np.ones(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
                self.hidden[l].set_extra_global_param("ImV", np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"], dtype=np.float32))
                self.hidden[l].set_extra_global_param("sNSum_all", np.zeros(p["N_BATCH"]))

            hidden_reset_params= {
                "V_reset": p["V_RESET"],
                "N_max_spike": p["N_MAX_SPIKE"],
                "N_neurons": p["NUM_HIDDEN"],
            }
            for l in range(p["N_HID_LAYER"]):
                hidden_reset_var_refs= {
                    "rp_ImV": genn_model.create_var_ref(self.hidden[l], "rp_ImV"),
                    "wp_ImV": genn_model.create_var_ref(self.hidden[l], "wp_ImV"),
                    "V": genn_model.create_var_ref(self.hidden[l], "V"),
                    "lambda_V": genn_model.create_var_ref(self.hidden[l], "lambda_V"),
                    "lambda_I": genn_model.create_var_ref(self.hidden[l], "lambda_I"),
                    "rev_t": genn_model.create_var_ref(self.hidden[l], "rev_t"),
                    "fwd_start": genn_model.create_var_ref(self.hidden[l], "fwd_start"),
                    "new_fwd_start": genn_model.create_var_ref(self.hidden[l], "new_fwd_start"),
                    "back_spike": genn_model.create_var_ref(self.hidden[l], "back_spike"),
                    "sNSum": genn_model.create_var_ref(self.hidden[l], "sNSum"),
                    "new_sNSum": genn_model.create_var_ref(self.hidden[l], "new_sNSum"),
                }
                print(f"Hidden layer {l} reset custom update: EVP_neuron_reset_reg_global")
                self.hidden_reset.append(self.model.add_custom_update("hidden_reset"+str(l), "neuronReset", EVP_neuron_reset_reg_global, hidden_reset_params, {}, hidden_reset_var_refs))
                self.hidden_reset[l].set_extra_global_param("sNSum_all", np.zeros(p["N_BATCH"]))

            if p["AVG_SNSUM"]:
                params= {"reduced_sNSum": 0.0}
                for l in range(p["N_HID_LAYER"]):
                    var_refs= {"sNSum": genn_model.create_var_ref(self.hidden[l], "sNSum")}
                    print(f"Hidden reg reduce {l} custom update: EVP_reg_reduce")
                    self.hidden_reg_reduce.append(self.model.add_custom_update("hidden_reg_reduce"+str(l), "sNSumReduce", EVP_reg_reduce, {}, params, var_refs))

                params= {"N_batch": p["N_BATCH"]}
                for l in range(p["N_HID_LAYER"]):
                    var_refs= {
                        "reduced_sNSum": genn_model.create_var_ref(self.hidden_reg_reduce[l], "reduced_sNSum"),
                        "sNSum": genn_model.create_var_ref(self.hidden[l], "sNSum")
                    }
                    print(f"Hidden SNSum apply {l} custom update: EVP_sNSum_apply")
                    self.hidden_redSNSum_apply.append(self.model.add_custom_update("hidden_redSNSum_apply"+str(l), "sNSumApply", EVP_sNSum_apply, params, {}, var_refs))


        # ----------------------------------------------------------------------------
        # output neuron initialisation
        # ----------------------------------------------------------------------------
        print("defining output ...")
        if p["LOSS_TYPE"][:-4] == "first_spike":
            output_params= {
                "tau_m": p["TAU_MEM_OUTPUT"],
                "tau_syn": p["TAU_SYN"],
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
            }
            if p["LOSS_TYPE"] == "first_spike":
                print("Output neurons: EVP_LIF_output_first_spike")
                self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_first_spike, output_params, self.output_init_vars)
            if p["LOSS_TYPE"] == "first_spike_exp":
                print("Output neurons: EVP_LIF_output_first_spike_exp")
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
            print("Output reset: EVP_neuron_reset_output_SHD_first_spike")
            self.output_reset= self.model.add_custom_update("output_reset", "neuronReset", EVP_neuron_reset_output_SHD_first_spike, output_reset_params, {}, output_reset_var_refs)


        if p["LOSS_TYPE"] == "max":
            output_params= {
                "tau_m": p["TAU_MEM_OUTPUT"],
                "tau_syn": p["TAU_SYN"],
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
            }
            print("Output neurons: EVP_LIF_output_max")
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
            print("Output reset: EVP_neuron_reset_output_SHD_max")
            self.output_reset= self.model.add_custom_update("output_reset", "neuronReset", EVP_neuron_reset_output_SHD_max, output_reset_params, {}, output_var_refs)


        if p["LOSS_TYPE"][:3] == "sum":
               
            output_params= {
                "tau_m": p["TAU_MEM_OUTPUT"],
                "tau_syn": p["TAU_SYN"],
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
            }
            if p["LOSS_TYPE"] == "sum":
                print("Output neurons: EVP_LIF_output_sum")
                the_output_neuron_type= EVP_LIF_output_sum
            if p["LOSS_TYPE"] == "sum_weigh_linear":
                print("Output neurons: EVP_LIF_output_sum_weigh_linear")
                the_output_neuron_type= EVP_LIF_output_sum_weigh_linear
            if p["LOSS_TYPE"] == "sum_weigh_exp":
                print("Output neurons: EVP_LIF_output_sum_weigh_exp")
                the_output_neuron_type= EVP_LIF_output_sum_weigh_exp
            if p["LOSS_TYPE"] == "sum_weigh_sigmoid":
                print("Output neurons: EVP_LIF_output_sum_weigh_sigmoid")
                the_output_neuron_type= EVP_LIF_output_sum_weigh_sigmoid
            if p["LOSS_TYPE"] == "sum_weigh_input":
                print("Output neurons: EVP_LIF_output_sum_weigh_input")
                the_output_neuron_type= EVP_LIF_output_sum_weigh_input
                output_params["N_neurons"]= self.num_output
                output_params["trial_steps"]= self.trial_steps
                self.output_init_vars["rp_V"]= self.trial_steps;
                self.output_init_vars["wp_V"]= 0;
                self.output_init_vars["avgInback"]= 0.0;
            self.output= self.model.add_neuron_population("output", self.num_output, the_output_neuron_type, output_params, self.output_init_vars)
            self.output.set_extra_global_param("label", np.zeros(self.data_max_length, dtype=np.float32)) # reserve space for labels
            if p["LOSS_TYPE"] == "sum_weigh_input":
                self.output.set_extra_global_param("aIbuf", np.zeros(p["N_BATCH"]*self.num_output*self.trial_steps*2, dtype=np.float32)) # reserve space for avgInput

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
            if p["LOSS_TYPE"] == "sum_weigh_input":
                output_reset_params["trial_steps"]= self.trial_steps
                output_var_refs["rp_V"]= genn_model.create_var_ref(self.output, "rp_V")
                output_var_refs["wp_V"]= genn_model.create_var_ref(self.output, "wp_V")
                print("Output reset custom update: EVP_neuron_reset_output_SHD_sum_weigh_input")
                self.output_reset= self.model.add_custom_update("output_reset", "neuronReset", EVP_neuron_reset_output_SHD_sum_weigh_input, output_reset_params, {}, output_var_refs)
            else:
                print("Output reset custom update: EVP_neuron_reset_output_SHD_sum")
                self.output_reset= self.model.add_custom_update("output_reset", "neuronReset", EVP_neuron_reset_output_SHD_sum, output_reset_params, {}, output_var_refs)


        if p["LOSS_TYPE"] == "avg_xentropy":
            output_params= {
                "tau_m": p["TAU_MEM_OUTPUT"],
                "tau_syn": p["TAU_SYN"],
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
            }
            print("Output neurons: EVP_LIF_output_SHD_avg_xentropy")
            self.output= self.model.add_neuron_population("output", self.num_output, EVP_LIF_output_SHD_avg_xentropy, output_params, self.output_init_vars)
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
            print("Output reset: EVP_neuron_reset_output_avg_xentropy")
            self.output_reset= self.model.add_custom_update("output_reset", "neuronReset", EVP_neuron_reset_output_avg_xentropy, output_reset_params, {}, output_var_refs)


        # ----------------------------------------------------------------------------
        # Optimiser initialisation
        # ----------------------------------------------------------------------------

        adam_params= {
            "beta1": p["ADAM_BETA1"],
            "beta2": p["ADAM_BETA2"],
            "epsilon": p["ADAM_EPS"],
            "tau_syn": p["TAU_SYN"],
        }
        self.adam_init_vars= {
            "m": 0.0,
            "v": 0.0,
        }
        adam_taum_params= {
            "beta1": p["ADAM_BETA1"],
            "beta2": p["ADAM_BETA2"],
            "epsilon": p["ADAM_EPS"],
            "min_tau_m": p["MIN_TAU_M"],
        }


        # ----------------------------------------------------------------------------
        # Synapse initialisation
        # ----------------------------------------------------------------------------
        print("defining synapses ...")

        self.in_to_hid_init_vars= {"dw": 0}
        self.in_to_hid_init_vars["w"]= genn_model.init_var(
            "Normal", {"mean": p["INPUT_HIDDEN_MEAN"], "sd": p["INPUT_HIDDEN_STD"]})

        self.hid_to_out_init_vars= {"dw": 0}
        self.hid_to_out_init_vars["w"]= genn_model.init_var(
            "Normal", {"mean": p["HIDDEN_OUTPUT_MEAN"], "sd": p["HIDDEN_OUTPUT_STD"]})

        if p["N_HID_LAYER"] > 1:
            self.hid_to_hidfwd_init_vars= {"dw": 0}
            self.hid_to_hidfwd_init_vars["w"]= genn_model.init_var(
                "Normal", {"mean": p["HIDDEN_HIDDENFWD_MEAN"], "sd": p["HIDDEN_HIDDENFWD_STD"]})
        if p["RECURRENT"]:
            self.hid_to_hid_init_vars= {"dw": 0}
            self.hid_to_hid_init_vars["w"]= genn_model.init_var(
                "Normal", {"mean": p["HIDDEN_HIDDEN_MEAN"], "sd": p["HIDDEN_HIDDEN_STD"]})

        print("in_to_hid synapses: EVP_input_synapse")
        self.in_to_hid= self.model.add_synapse_population(
            "in_to_hid", "DENSE_INDIVIDUALG", NO_DELAY, self.input, self.hidden[0], EVP_input_synapse,
            {}, self.in_to_hid_init_vars, {}, {}, my_Exp_Curr, {"tau": p["TAU_SYN"]}, {})

        print("hid_to_out synapses: EVP_synapse")
        self.hid_to_out= self.model.add_synapse_population(
            "hid_to_out", "DENSE_INDIVIDUALG", NO_DELAY, self.hidden[-1], self.output, EVP_synapse,
            {}, self.hid_to_out_init_vars, {}, {}, my_Exp_Curr, {"tau": p["TAU_SYN"]}, {})

        for l in range(p["N_HID_LAYER"]-1):
            print(f"hid_to_hidfwd synapses {l} to {l+1}: EVP_synapse")
            self.hid_to_hidfwd.append(self.model.add_synapse_population(
                "hid_to_hidfwd"+str(l), "DENSE_INDIVIDUALG", NO_DELAY, self.hidden[l], self.hidden[l+1], EVP_synapse,
                {}, self.hid_to_hidfwd_init_vars, {}, {}, my_Exp_Curr, {"tau": p["TAU_SYN"]}, {}))
        
        if p["RECURRENT"]:
            for l in range(p["N_HID_LAYER"]):
                print(f"hid_to_hid synapses {l} to {l}: EVP_synapse")
                self.hid_to_hid.append(self.model.add_synapse_population(
                    "hid_to_hid"+str(l), "DENSE_INDIVIDUALG", NO_DELAY, self.hidden[l], self.hidden[l], EVP_synapse,
                    {}, self.hid_to_hid_init_vars, {}, {}, my_Exp_Curr, {"tau": p["TAU_SYN"]}, {}))

        self.optimisers= []
        # learning updates for synapses
        var_refs = {"dw": genn_model.create_wu_var_ref(self.in_to_hid, "dw")}
        print("in_to_hid reduce: EVP_grad_reduce")
        self.in_to_hid_reduce= self.model.add_custom_update(
            "in_to_hid_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
        var_refs = {"gradient": genn_model.create_wu_var_ref(self.in_to_hid_reduce, "reduced_dw"),
                    "variable": genn_model.create_wu_var_ref(self.in_to_hid, "w")}
        print("in_to_hid learn: adam_optimizer_model")
        self.in_to_hid_learn= self.model.add_custom_update(
            "in_to_hid_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)
        self.optimisers.append(self.in_to_hid_learn)
        
        var_refs = {"dw": genn_model.create_wu_var_ref(self.hid_to_out, "dw")}
        print("hid_to_out reduce: EVP_grad_reduce")
        self.hid_to_out_reduce= self.model.add_custom_update(
            "hid_to_out_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
        var_refs = {"gradient": genn_model.create_wu_var_ref(self.hid_to_out_reduce, "reduced_dw"),
                    "variable": genn_model.create_wu_var_ref(self.hid_to_out, "w")}
        print("hid_to_out learn: adam_optimizer_model")
        self.hid_to_out_learn= self.model.add_custom_update(
            "hid_to_out_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)
        self.hid_to_out.pre_target_var= "revIsyn"
        self.optimisers.append(self.hid_to_out_learn)

        for l in range(p["N_HID_LAYER"]-1):
            var_refs = {"dw": genn_model.create_wu_var_ref(self.hid_to_hidfwd[l], "dw")}
            print(f"hid_to_hidfwd reduce {l}: EVP_grad_reduce")
            self.hid_to_hidfwd_reduce.append(self.model.add_custom_update(
                "hid_to_hidfwd_reduce"+str(l),"EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs))
            var_refs = {"gradient": genn_model.create_wu_var_ref(self.hid_to_hidfwd_reduce[l], "reduced_dw"),
                        "variable": genn_model.create_wu_var_ref(self.hid_to_hidfwd[l], "w")}
            print(f"hid_to_hidfwd learn {l}: adam_optimizer_model")            
            self.hid_to_hidfwd_learn.append(self.model.add_custom_update(
                "hid_to_hidfwd_learn"+str(l),"EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs))
            self.hid_to_hidfwd[l].pre_target_var= "revIsyn"
            self.optimisers.append(self.hid_to_hidfwd_learn[l])            

        if p["RECURRENT"]:
            for l in range(p["N_HID_LAYER"]):
                var_refs = {"dw": genn_model.create_wu_var_ref(self.hid_to_hid[l], "dw")}
                print(f"hid_to_hid reduce: EVP_grad_reduce")
                self.hid_to_hid_reduce.append(self.model.add_custom_update(
                    "hid_to_hid_reduce"+str(l),"EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs))
                var_refs = {"gradient": genn_model.create_wu_var_ref(self.hid_to_hid_reduce[l], "reduced_dw"),
                            "variable": genn_model.create_wu_var_ref(self.hid_to_hid[l], "w")}
                print(f"hid_to_hid learn: adam_optimizer_model")
                self.hid_to_hid_learn.append(self.model.add_custom_update(
                    "hid_to_hid_learn"+str(l),"EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs))
                self.hid_to_hid[l].pre_target_var= "revIsyn"
                self.optimisers.append(self.hid_to_hid_learn[l])
        # learning updates for taum
        if p["TRAIN_TAUM"]:
            for l in range(p["N_HID_LAYER"]):            
                var_refs = {"dw": genn_model.create_var_ref(self.hidden[l], "dtaum")}
                print(f"Hidden taum {l} reduce: EVP_grad_reduce")
                self.hidden_taum_reduce.append(self.model.add_custom_update(
                    "hidden_taum_reduce"+str(l),"EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs))
                var_refs = {"gradient": genn_model.create_var_ref(self.hidden_taum_reduce[l], "reduced_dw"),
                            "variable": genn_model.create_var_ref(self.hidden[l], "tau_m")}
                print(f"Hidden taum {l} learn: adam_optimizer_model_taum")
                self.hidden_taum_learn.append(self.model.add_custom_update(
                    "hidden_taum_learn"+str(l),"EVPLearn", adam_optimizer_model_taum, adam_taum_params, self.adam_init_vars, var_refs))
                self.optimisers.append(self.hidden_taum_learn[l])
           
        # DEBUG hidden layer spike numbers
        if p["DEBUG_HIDDEN_N"]:
            if p["REG_TYPE"] != "Thomas1":
                for l in range(p["N_HID_LAYER"]):            
                    self.model.neuron_populations["hidden"+str(l)].spike_recording_enabled= True

        # enable buffered spike recording where desired
        for pop in p["REC_SPIKES"]:
            self.model.neuron_populations[pop].spike_recording_enabled= True

        # add an input accumulator neuron and wire it up
        if p["LOSS_TYPE"] == "sum_weigh_input":
            accumulator_params= {
                "tau_m": p["TAU_ACCUMULATOR"],
            }
            accumulator_init_vars= {
                "V": 0.0,
            }
            print("input accumulator neuron: EVP_LIF_input_accumulator")
            self.accumulator= self.model.add_neuron_population("accumulator", 1, EVP_LIF_input_accumulator, accumulator_params, accumulator_init_vars)
            print("input_accumulator synapses: 'StaticPulse', 'DeltaCurr'")
            self.in_to_acc= self.model.add_synapse_population("in_to_acc", "DENSE_GLOBALG", NO_DELAY, self.input, self.accumulator, "StaticPulse",
                {}, {"g": 0.01}, {}, {}, "DeltaCurr", {}, {})
            print("accumulator_output synapses: EVP_accumulator_output_synapse")
            self.acc_to_out= self.model.add_synapse_population("acc_to_out", "DENSE_GLOBALG", NO_DELAY, self.accumulator, self.output, EVP_accumulator_output_synapse,
                {}, {}, {}, {}, "DeltaCurr", {}, {})
            self.acc_to_out.ps_target_var="avgIn"
            accumulator_reset_var_refs= {
                "V": genn_model.create_var_ref(self.accumulator, "V"),
            }
            print("accumulator_reset: EVP_neuron_reset_input_accumulator")
            self.accumulator_reset= self.model.add_custom_update("accumulator_reset", "neuronReset", EVP_neuron_reset_input_accumulator, {}, {}, accumulator_reset_var_refs)
        print("model definition complete ...")

        
    """
    ----------------------------------------------------------------------------
    Helpers to run the model
    ----------------------------------------------------------------------------
    """

    def zero_insyn(self, p):
        self.in_to_hid.in_syn[:]= 0.0
        self.in_to_hid.push_in_syn_to_device()
        self.hid_to_out.in_syn[:]= 0.0
        self.hid_to_out.push_in_syn_to_device()
        for l in range(p["N_HID_LAYER"]-1):
            self.hid_to_hidfwd[l].in_syn[:]= 0.0
            self.hid_to_hidfwd[l].push_in_syn_to_device()
        if p["RECURRENT"]:
            for l in range(p["N_HID_LAYER"]):
                self.hid_to_hid[l].in_syn[:]= 0.0
                self.hid_to_hid[l].push_in_syn_to_device()

    def calc_balance(self,Y_t, Z_t, Y_e):
        if Z_t is not None:
            speakers= set(Z_t)
            print("train:")
            sn= []
            for s in speakers:
                sn.append([ np.sum(np.logical_and(Y_t == d, Z_t == s)) for d in range(20) ])
                print(f"Speaker {s}: {sn[-1]}")
            sn= np.array(sn)
            snm= np.sum(sn, axis=0)
            print(f"Sum across speakers: {snm}")
            print("eval:")
            sne= [ np.sum(Y_e == d) for d in range(20) ]
            print(sne)
        else:
            snm= []
            sne= []
        return (snm, sne)


    def trial_setup(self, X_train, Y_train, Z_train, X_eval, Y_eval, snm, sne, p):
        N_trial= 0
        if X_train is not None:
            assert(Y_train is not None)
            if "blend" in p["AUGMENTATION"]:
                N_train= p["N_TRAIN"]
            else:
                if p["BALANCE_TRAIN_CLASSES"]:
                    N_train= np.min(snm)*self.N_class
                else:
                    N_train= len(X_train)
                
            N_trial_train= (N_train-1) // p["N_BATCH"] + 1  # large enough to fit potentially incomplete last batch
            N_trial+= N_trial_train
        else:
            N_trial_train= 0
        if X_eval is not None:
            assert(Y_eval is not None)
            if p["BALANCE_EVAL_CLASSES"]:
                N_eval= np.min(sne)*self.N_class
            else:
                N_eval= len(X_eval)
            N_trial_eval= (N_eval-1) // p["N_BATCH"] + 1  # large enough to fit potentially incomplete last batch
            N_trial+= N_trial_eval
        else:
            N_trial_eval= 0
        print(f"N_trial_train= {N_trial_train}, N_trial_eval= {N_trial_eval}")
        return (N_train, N_trial_train, N_trial_eval, N_trial)

    def balance_classes(self, X, Y, Z, X_e, Y_e, snm, sne, p):
        if p["BALANCE_TRAIN_CLASSES"]:
            ncl= np.min(snm)
            newX= []
            newY= []
            newZ= []
            for c in range(self.N_class):
                which= Y == c
                cX= X[which]
                cY= Y[which]
                cZ= Z[which]
                ids= np.arange(len(cX),dtype= int)
                self.datarng.shuffle(ids)
                cX= cX[ids]
                cY= cY[ids]
                cZ= cZ[ids]
                newX.append(cX[:ncl])
                newY.append(cY[:ncl])
                newZ.append(cZ[:ncl])
            newX= np.hstack(newX)
            newY= np.hstack(newY)
            newZ= np.hstack(newZ)
        else:
            newX= X
            newY= Y
            newZ= Z
        print(newX.shape)
        if p["BALANCE_EVAL_CLASSES"]:
            ncl= np.min(sne)
            newX_e= []
            newY_e= []
            for c in range(self.N_class):
                which= Y_e == c
                cX= X_e[which]
                cY= Y_e[which]
                ids= np.arange(len(cX),dtype= int)
                self.datarng.shuffle(ids)
                cX= cX[ids]
                cY= cY[ids]
                newX_e.append(cX[:ncl])
                newY_e.append(cY[:ncl])
            newX_e= np.hstack(newX_e)
            newY_e= np.hstack(newY_e)
        else:
            newX_e= X_e
            newY_e= Y_e
        print(newX_e.shape)
        return (newX, newY, newZ, newX_e, newY_e)
    """
    ----------------------------------------------------------------------------
    Run the model
    ----------------------------------------------------------------------------
    """
            
    def run_model(self, number_epochs, p, shuffle, X_train= None, Y_train= None, Z_train= None, X_eval= None, Y_eval= None, resfile= None):      
        if p["LOAD_LAST"]:
            self.in_to_hid.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_input_hidden_last.npy"))
            self.in_to_hid.push_var_to_device("w")
            self.hid_to_out.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_output_last.npy"))
            self.hid_to_out.push_var_to_device("w")
            for l in range(p["N_HID_LAYER"]-1):
                self.hid_to_hidfwd[l].vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden"+str(l)+"_hidden"+str(l+1)+"_last.npy"))
                self.hid_to_hidfwd[l].push_var_to_device("w")
            if p["RECURRENT"]:
                for l in range(p["N_HID_LAYER"]):
                    self.hid_to_hid[l].vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_hidden"+str(l)+"_last.npy"))
                    self.hid_to_hid[l].push_var_to_device("w")

        # set up run
        snm, sne= self.calc_balance(Y_train, Z_train, Y_eval)
        N_train, N_trial_train, N_trial_eval, N_trial= self.trial_setup(X_train, Y_train, Z_train, X_eval, Y_eval, snm, sne, p)
        
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

        if len(p["AUGMENTATION"]) == 0:
            # build and assign the input spike train and corresponding labels
            # these are padded to multiple of batch size for both train and eval portions
            if p["BALANCE_TRAIN_CLASSES"] or p["BALANCE_EVAL_CLASSES"]:
                X_train, Y_train, Z_train, X_eval, Y_eval= self.balance_classes(X_train, Y_train, Z_train, X_eval, Y_eval, snm, sne, p)
            X, Y, input_start, input_end= self.generate_input_spiketimes_shuffle_fast(p, X_train, Y_train, X_eval, Y_eval)
            self.input.extra_global_params["spikeTimes"].view[:len(X)]= X
            self.input.push_extra_global_param_to_device("spikeTimes")
            self.input_set.extra_global_params["allStartSpike"].view[:len(input_start)]= input_start
            self.input_set.push_extra_global_param_to_device("allStartSpike")
            self.input_set.extra_global_params["allEndSpike"].view[:len(input_end)]= input_end
            self.input_set.push_extra_global_param_to_device("allEndSpike")
        if X_train is not None:
            input_id= np.arange(N_train)
        else:
            input_id= []
        all_input_id= np.arange(N_trial*p["N_BATCH"])
        self.input_set.extra_global_params["allInputID"].view[:len(all_input_id)]= all_input_id
        self.input_set.push_extra_global_param_to_device("allInputID")

        if p["COLLECT_CONFUSION"]:
            confusion= {
                "train": [],
                "eval": []
            }
        if p["REC_PREDICTIONS"]:
            all_predict= {
                "train": [],
                "eval": []
            }
            all_label= {
                "train": [],
                "eval": []
            }
        correctEMA= 0       # exponential moving average of evaluation correct (fast)
        correctEMAslow= 0   # exponential moving average of evaluation correct (slow)
        red_lr_last= 0      # epoch when LR was last reduced
        for epoch in range(number_epochs):
            # if we are doing augmentation, the entire spike time array needs to be set up anew.
            lX= copy.deepcopy(X_train)
            lY= copy.deepcopy(Y_train)
            lZ= copy.deepcopy(Z_train)
            lX_eval= copy.deepcopy(X_eval)
            lY_eval= copy.deepcopy(Y_eval)
            if p["BALANCE_TRAIN_CLASSES"] or p["BALANCE_EVAL_CLASSES"]:
                lX, lY, lZ, lX_eval, lY_eval= self.balance_classes(lX, lY, lZ, lX_eval, lY_eval, snm, sne, p)
            if ("NORMALISE_SPIKE_NUMBER" in p["AUGMENTATION"]) and (p["AUGMENTATION"]["NORMALISE_SPIKE_NUMBER"]):
                lX, lX_eval= self.normalise_spike_number(lX, lX_eval)
            if N_trial_train > 0 and len(p["AUGMENTATION"]) > 0:
                for aug in p["AUGMENTATION"]:
                    if aug == "blend":
                        lX, lY, lZ= blend_dataset(lX,lY,lZ,self.datarng, p["AUGMENTATION"][aug],p)
                    if aug == "random_shift":
                        lX= random_shift(lX,self.datarng, p["AUGMENTATION"][aug],p)
                    if aug == "random_dilate":
                        lX= random_dilate(lX,self.datarng, p["AUGMENTATION"][aug][0], p["AUGMENTATION"][aug][1],p)
                    if aug == "ID_jitter":
                        lX= ID_jitter(lX,self.datarng, p["AUGMENTATION"][aug],p)
                X, Y, input_start, input_end= self.generate_input_spiketimes_shuffle_fast(p, lX, lY, lX_eval, lY_eval)
                self.input.extra_global_params["spikeTimes"].view[:len(X)]= X
                self.input.push_extra_global_param_to_device("spikeTimes")
                self.input_set.extra_global_params["allStartSpike"].view[:len(input_start)]= input_start
                self.input_set.push_extra_global_param_to_device("allStartSpike")
                self.input_set.extra_global_params["allEndSpike"].view[:len(input_end)]= input_end
                self.input_set.push_extra_global_param_to_device("allEndSpike")

            if N_trial_train > 0 and shuffle:
                # by virtue of input_id being the right length we do not shuffle
                # padding inputs
                self.datarng.shuffle(input_id)
                all_input_id[:len(input_id)]= input_id
                Y[:len(input_id)]= lY[input_id]
                self.output.extra_global_params["label"].view[:len(Y)]= Y
                self.output.push_extra_global_param_to_device("label")
                self.input_set.extra_global_params["allInputID"].view[:len(all_input_id)]= all_input_id
                self.input_set.push_extra_global_param_to_device("allInputID")

            if p["REC_PREDICTIONS"]:
                predict= {
                    "train": [],
                    "eval": []
                }
                label= {
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
            for l in range(p["N_HID_LAYER"]):
                for var, val in self.hidden_init_vars.items():
                    if var != "tau_m":               # do not reset tau_m at beginning of epoch
                        self.hidden[l].vars[var].view[:]= val
                    else:
                        self.hidden[l].pull_var_from_device(var)
                self.hidden[l].push_state_to_device()
                self.hidden[l].extra_global_params["pDrop"].view[:]= p["PDROP_HIDDEN"] 
                if p["HIDDEN_NOISE"] > 0.0:
                    self.hidden[l].extra_global_params["A_noise"].view[:]= p["HIDDEN_NOISE"]
            for var, val in self.output_init_vars.items():
                self.output.vars[var].view[:]= val
            self.output.push_state_to_device()
            self.model.custom_update("EVPReduce")  # this zeros dw (so as to ignore eval gradients from last epoch!)

            if p["DEBUG_HIDDEN_N"]:
                all_hidden_n= [[] for _ in range(p["N_HID_LAYER"])]
                all_sNSum= [[] for _ in range(p["N_HID_LAYER"])]

            rewire_sNSum= [[] for _ in range(p["N_HID_LAYER"])]

            if p["COLLECT_CONFUSION"]:
                conf= {
                    "train": np.zeros((self.N_class,self.N_class)),
                    "eval": np.zeros((self.N_class,self.N_class))
                }

            spk_rec_offset= 0
            for trial in range(N_trial):
                trial_end= (trial+1)*p["TRIAL_MS"]

                # assign the input spike train and corresponding labels
                if trial < N_trial_train:
                    phase= "train"
                    # set the actual batch size
                    if trial == N_trial_train-1:
                        N_batch= len(lX)-(N_trial_train-1)*p["N_BATCH"]
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
                    for l in range(p["N_HID_LAYER"]): 
                        self.hidden[l].extra_global_params["pDrop"].view[:]= 0.0
                        if p["HIDDEN_NOISE"] > 0.0:
                            self.hidden[l].extra_global_params["A_noise"].view[:]= 0.0
                self.input_set.extra_global_params["trial"].view[:]= trial
                self.model.custom_update("inputUpdate")
                self.input.extra_global_params["t_offset"].view[:]= self.model.t

                if p["DEBUG_HIDDEN_N"]:
                    if p["REG_TYPE"] != "Thomas1":
                        spike_N_hidden= []
                        for l in range(p["N_HID_LAYER"]):
                            spike_N_hidden.append(np.zeros(N_batch))

                int_t= 0
                while (self.model.t < trial_end-1e-1*p["DT_MS"]):
                    self.model.step_time()
                    int_t += 1

                    # DEBUG of middle layer activity
                    if p["DEBUG_HIDDEN_N"]:
                        if int_t%p["SPK_REC_STEPS"] == 0:
                            if p["REG_TYPE"] != "Thomas1":
                                self.model.pull_recording_buffers_from_device()
                                for l in range(p["N_HID_LAYER"]):
                                    x= self.model.neuron_populations["hidden"+str(l)].spike_recording_data
                                    for btch in range(N_batch):
                                        spike_N_hidden[l][btch]+= len(x[btch][0])

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
                        self.zero_insyn(p)
                            
                # at end of trial
                spk_rec_offset+= N_batch-1  # the -1 for compensating the normal progression of time

                # do not learn after the 0th trial where lambdas are meaningless
                if (phase == "train") and trial > 0:
                    update_adam(learning_rate, adam_step, self.optimisers)
                    adam_step += 1
                    self.model.custom_update("EVPReduce")
                    #if trial%2 == 1:
                    self.model.custom_update("EVPLearn")

                self.zero_insyn(p)
                
                if p["REG_TYPE"] == "Thomas1":
                    # for hidden regularistation prepare "sNSum_all"
                    for l in range(p["N_HID_LAYER"]):
                        self.hidden_reset[l].extra_global_params["sNSum_all"].view[:]= np.zeros(p["N_BATCH"])
                        self.hidden_reset[l].push_extra_global_param_to_device("sNSum_all")

                if p["LOSS_TYPE"][:-4] == "first_spike":
                    # need to copy new_first_spike_t from device before neuronReset!
                    self.output.pull_var_from_device("new_first_spike_t")
                    nfst= self.output.vars["new_first_spike_t"].view[:N_batch,:self.N_class].copy()
                    # neurons that did not spike set to spike time in the future
                    nfst[nfst < 0.0]= self.model.t + 1.0
                    pred= np.argmin(nfst, axis=-1)

                if p["LOSS_TYPE"] == "avg_xentropy":
                    # need to copy sum_V and loss from device before neuronReset!
                    self.output.pull_var_from_device("sum_V")
                    self.output.pull_var_from_device("loss")

                if p["LOSS_TYPE"][:3] == "sum":
                    # do the custom updates for softmax!
                    self.model.custom_update("Softmax1")
                    self.model.custom_update("Softmax2")
                    self.model.custom_update("Softmax3")
                    #self.output.pull_var_from_device("sum_V")
                          
                self.model.custom_update("neuronReset")

                if (p["REG_TYPE"] == "simple" or p["REG_TYPE"] == "Thomas1") and p["AVG_SNSUM"]:
                    self.model.custom_update("sNSumReduce")
                    self.model.custom_update("sNSumApply")

                if p["REG_TYPE"] == "Thomas1": 
                    for l in range(p["N_HID_LAYER"]):
                        self.hidden_reset[l].pull_extra_global_param_from_device("sNSum_all")
                        self.hidden[l].extra_global_params["sNSum_all"].view[:]= self.hidden_reset.extra_global_params["sNSum_all"].view[:]
                        self.hidden[l].push_extra_global_param_to_device("sNSum_all")
                        if p["DEBUG_HIDDEN_N"]:
                            spike_N_hidden[l]= self.hidden_reset[l].extra_global_params["sNSum_all"].view[:N_batch].copy()

                # collect data for rewiring rule for silent neurons
                for l in range(p["N_HID_LAYER"]):
                    self.hidden[l].pull_var_from_device("sNSum")
                    rewire_sNSum[l].append(np.sum(self.hidden[l].vars["sNSum"].view.copy(),axis= 0))

                # record training loss and error
                # NOTE: the neuronReset does the calculation of expsum and updates exp_V for loss types sum and max
                if p["LOSS_TYPE"] == "max":
                    self.output.pull_var_from_device("exp_V")
                    pred= np.argmax(self.output.vars["exp_V"].view[:N_batch,:self.N_class], axis=-1)
                if p["LOSS_TYPE"][:3] == "sum":
                    self.output.pull_var_from_device("SoftmaxVal")
                    #if phase == "eval":
                    #    for i in range(N_batch):
                    #        print(f"{np.argmax(self.output.vars['sum_V'].view[i,:self.N_class])},{np.max(self.output.vars['sum_V'].view[i,:self.N_class])}")
                    #        print(f"{np.argmax(self.output.vars['SoftmaxVal'].view[i,:self.N_class])},{np.max(self.output.vars['SoftmaxVal'].view[i,:self.N_class])}")
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

                if p["COLLECT_CONFUSION"]:
                    for pr, lb in zip(pred,lbl):
                        conf[phase][pr,lb]+= 1

                if p["LOSS_TYPE"][:-4] == "first_spike":
                    self.output.pull_var_from_device("expsum")
                    self.output.pull_var_from_device("exp_st")
                    if p["LOSS_TYPE"] == "first_spike":
                        losses= self.loss_func_first_spike(nfst, lbl, trial, self.N_class, N_batch)
                    if p["LOSS_TYPE"] == "first_spike_exp":
                        losses= self.loss_func_first_spike_exp(nfst, lbl, trial, self.N_class, N_batch)

                if p["LOSS_TYPE"] == "max":
                    self.output.pull_var_from_device("expsum")
                    losses= self.loss_func_max(lbl, N_batch)   # uses self.output.vars["exp_V"].view and self.output.vars["expsum"].view

                if p["LOSS_TYPE"][:3] == "sum":
                    losses= self.loss_func_sum(lbl, N_batch)   # uses self.output.vars["SoftmaxVal"].view 

                if p["LOSS_TYPE"] == "avg_xentropy":
                    losses= self.loss_func_avg_xentropy(lbl, N_batch)   # uses self.output.vars["loss"].view

                #with open("debug.txt","a") as f:
                #    for i in range(len(lbl)):
                #        f.write(f"{lbl[i]}\n")
                good[phase] += np.sum(pred == lbl)
                #if phase == "eval":
                #    print(pred)
                #    print(lbl)
                #xx= pred == lbl
                #print(xx.astype(int))
                if p["REC_PREDICTIONS"]:
                    predict[phase].append(pred)
                    label[phase].append(lbl)
                    
                the_loss[phase].append(losses)

                if p["DEBUG_HIDDEN_N"]:
                    for l in range(p["N_HID_LAYER"]):
                        all_hidden_n[l].append(spike_N_hidden[l])
                        self.hidden[l].pull_var_from_device("sNSum")
                        all_sNSum[l].append(np.mean(self.hidden[l].vars['sNSum'].view[:N_batch],axis= 0))

                if ([epoch, trial] in p["W_OUTPUT_EPOCH_TRIAL"]): 
                    self.in_to_hid.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_input_hidden_e{}_t{}.npy".format(epoch,trial)), self.in_to_hid.vars["w"].view.copy())
                    self.hid_to_out.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_output_e{}_t{}.npy".format(epoch,trial)), self.hid_to_out.vars["w"].view.copy())
                    for l in range(p["N_HID_LAYER"]-1):
                        self.hid_to_hidfwd[l].pull_var_from_device("w")
                        np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden"+str(l)+"_hidden"+str(l+1)+"_e{}_t{}.npy".format(epoch,trial)), self.hid_to_hidfwd[l].vars["w"].view.copy())
                       
                    if p["RECURRENT"]:
                        for l in range(p["N_HID_LAYER"]):
                            self.hid_to_hid[l].pull_var_from_device("w")
                            np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_hidden"+str(l)+"_e{}_t{}.npy".format(epoch,trial)), self.hid_to_hid[l].vars["w"].view.copy())
                if p["TRAIN_TAUM"]:
                    if ([epoch,trial] in p["TAUM_OUTPUT_EPOCH_TRIAL"]):
                        for l in range(p["N_HID_LAYER"]):
                            self.hidden[l].pull_var_from_device("tau_m")
                            np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_tau_m_hidden"+str(l)+"_e{}_t{}.npy".format(epoch,trial)), self.hidden[l].vars["tau_m"].view.copy())

            if N_trial_train > 0:
                correct= good["train"]/len(lX)
            else:
                correct= 0
            if N_trial_eval > 0:
                correct_eval= good["eval"]/len(lX_eval)
            else:
                correct_eval= 0

            n_silent= []
            for l in range(p["N_HID_LAYER"]):
                rewire_sNSum[l]= np.sum(np.array(rewire_sNSum[l]),axis= 0)
                silent= rewire_sNSum[l] == 0
                n_silent.append(np.sum(silent))
                if p["REWIRE_SILENT"]:
                    # rewire input to hidden or hidden to hidden fwd
                    if l == 0:
                        pop= self.in_to_hid
                    else:
                        pop= self.hid_to_hidfwd[l-1]
                        
                    pop.pull_var_from_device("w")
                    ith_w= pop.vars["w"].view[:]
                    if p["REWIRE_LIFT"] != 0.0:
                        ith_w+= p["REWIRE_LIFT"]
                    else:
                        ith_w.shape= (self.num_input*(p["N_INPUT_DELAY"]+1),p["NUM_HIDDEN"])
                        n_new= self.num_input*(p["N_INPUT_DELAY"]+1)*n_silent[l]
                        ith_w[:,silent]= np.reshape(rng.standard_normal(n_new)*p["INPUT_HIDDEN_STD"]+p["INPUT_HIDDEN_MEAN"], (self.num_input*(p["N_INPUT_DELAY"]+1), n_silent[l]))
                    pop.push_var_to_device("w")
                        
            if p["DEBUG_HIDDEN_N"]:
                for l in range(p["N_HID_LAYER"]):
                    all_hidden_n[l]= np.hstack(all_hidden_n[l])
                    all_sNSum[l]= np.hstack(all_sNSum[l])
                    print("Hidden spikes "+str(l)+" in model per trial: {} +/- {}, min {}, max {}".format(np.mean(all_hidden_n[l]),np.std(all_hidden_n[l]),np.amin(all_hidden_n[l]),np.amax(all_hidden_n[l])))
                    print("Hidden spikes "+str(l)+" per trial per neuron across batches: {} +/- {}, min {}, max {}".format(np.mean(all_sNSum[l]),np.std(all_sNSum[l]),np.amin(all_sNSum[l]),np.amax(all_sNSum[l])))

            print("{} Training Correct: {}, Training Loss: {}, Evaluation Correct: {}, Evaluation Loss: {}, Silent: {}".format(epoch, correct, np.mean(the_loss["train"]), correct_eval, np.mean(the_loss["eval"]), n_silent))
            print(f"Training examples: {len(lX)}, Evaluation examples: {len(lX_eval)}")
            
            
            if resfile is not None:
                resfile.write("{} {} {} {} {}".format(epoch, correct, np.mean(the_loss["train"]), correct_eval, np.mean(the_loss["eval"])))
                if p["DEBUG_HIDDEN_N"]:
                    for l in range(p["N_HID_LAYER"]):
                        resfile.write(" {} {} {} {}".format(np.mean(all_hidden_n[l]),np.std(all_hidden_n[l]),np.amin(all_hidden_n[l]),np.amax(all_hidden_n[l])))
                        resfile.write(" {} {} {} {} {}".format(np.mean(all_sNSum[l]),np.std(all_sNSum[l]),np.amin(all_sNSum[l]),np.amax(all_sNSum[l]),n_silent[l]))
                resfile.write("\n")
                resfile.flush()

            # learning rate schedule depending on EMA of performance
            correctEMA= p["EMA_ALPHA1"]*correctEMA+(1.0-p["EMA_ALPHA1"])*correct_eval
            correctEMAslow= p["EMA_ALPHA2"]*correctEMAslow+(1.0-p["EMA_ALPHA2"])*correct_eval
            if (epoch-red_lr_last > p["MIN_EPOCH_ETA_FIXED"]) and (correctEMA <= correctEMAslow):
                learning_rate*= p["ETA_FAC"]
                red_lr_last= epoch
                print("EMA {}, EMAslow {}, Reduced LR to {}".format(correctEMA, correctEMAslow, learning_rate))
                print(learning_rate)
            print(f"EMA: {correctEMA}, EMA_slow: {correctEMAslow}")
            if p["REC_PREDICTIONS"]:
                predict[phase]= np.hstack(predict[phase])
                label[phase]= np.hstack(label[phase])
                all_predict[phase].append(predict[phase])
                all_label[phase].append(label[phase])
          
            if p["COLLECT_CONFUSION"]:
                for ph in ["train","eval"]:
                    confusion[ph].append(conf[ph])

        for pop in p["REC_SPIKES"]:
            spike_t[pop]= np.hstack(spike_t[pop])
            spike_ID[pop]= np.hstack(spike_ID[pop])

        for rec_t, rec_var, rec_list in [ (rec_n_t, rec_vars_n, p["REC_NEURONS"]), (rec_s_t, rec_vars_s, p["REC_SYNAPSES"])]:
            for pop, var in rec_list:
                rec_var[var+pop]= np.array(rec_var[var+pop])
                print(rec_var[var+pop].shape)
            rec_t= np.array(rec_t)

        rec_spk_lbl= np.array(rec_spk_lbl)
        rec_spk_pred= np.array(rec_spk_pred)
        rec_n_lbl= np.array(rec_n_lbl)
        rec_n_pred= np.array(rec_n_pred)
        rec_s_lbl= np.array(rec_s_lbl)
        rec_s_pred= np.array(rec_s_pred)

        if p["COLLECT_CONFUSION"]:
            for ph in ["train","eval"]:
                confusion[ph]= np.array(confusion[ph])

        if p["REC_PREDICTIONS"]:
            for ph in ["train","eval"]:
                if len(all_predict[ph]) > 0:
                    all_predict[ph]= np.vstack(all_predict[ph])
                if len(all_label[ph]) > 0:
                    all_label[ph]= np.vstack(all_label[ph])
                
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

            if p["COLLECT_CONFUSION"]:
                for ph in ["train","eval"]:
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_confusion_"+ph), confusion[ph])

            if p["REC_PREDICTIONS"]:
                for ph in ["train","eval"]:
                    if len(all_predict[ph]) > 0:
                        np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_predictions_"+ph), all_predict[ph])
                    if len(all_label[ph]) > 0:
                        np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_labels_"+ph), all_label[ph])
                    
                
        # Saving results
        if p["WRITE_TO_DISK"]:
            self.in_to_hid.pull_var_from_device("w")
            np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_input_hidden_last.npy"), self.in_to_hid.vars["w"].view)
            self.hid_to_out.pull_var_from_device("w")
            np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_output_last.npy"), self.hid_to_out.vars["w"].view)
            for l in range(p["N_HID_LAYER"]-1):
                self.hid_to_hidfwd[l].pull_var_from_device("w")
                np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden"+str(l)+"_hidden"+str(l+1)+"_last.npy"), self.hid_to_hidfwd[l].vars["w"].view)
                
            if p["RECURRENT"]:
                for l in range(p["N_HID_LAYER"]):
                    self.hid_to_hid[l].pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], p["NAME"]+"_w_hidden_hidden"+str(l)+"_last.npy"), self.hid_to_hid[l].vars["w"].view)

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
        if p["N_VALIDATE"] > 0:
            if p["EVALUATION"] == "random":
                X_train, Y_train, X_eval, Y_eval= self.split_SHD_random(self.X_train_orig, self.Y_train_orig, p)
            if p["EVALUATION"] == "speaker":
                X_train, Y_train, Z_train, X_eval, Y_eval, Z_eval= self.split_SHD_speaker(self.X_train_orig, self.Y_train_orig, self.Z_train_orig, p["SPEAKER_LEFT"], p)
            if p["EVALUATION"] == "validation_set":
                X_train= self.X_train_orig
                Y_train= self.Y_train_orig
                Z_train= self.Z_train_orig
                X_eval= self.X_eval_orig
                Y_eval= self.Y_eval_orig
                Z_eval= self.Z_eval_orig
        else:
            X_train= self.X_train_orig
            Y_train= self.Y_train_orig
            Z_train= self.Z_train_orig
            X_eval= None
            Y_eval= None
            Z_eval= None
        return self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_train= X_train, Y_train= Y_train, Z_train= Z_train, X_eval= X_eval, Y_eval= Y_eval, resfile= resfile)
        
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
            X_train, Y_train, Z_train, X_eval, Y_eval, Z_eval= self.split_SHD_speaker(self.X_train_orig, self.Y_train_orig, self.Z_train_orig, i, p)
            res= self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_train= X_train, Y_train= Y_train, Z_train= Z_train, X_eval= X_eval, Y_eval= Y_eval, resfile= resfile)
            all_res.append([ res[4], res[5] ])
            times.append(perf_counter()-start_t)
        return all_res, times
    
    def test(self, p):
        self.define_model(p, False)
        if p["BUILD"]:
            self.model.build()
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        return self.run_model(1, p, False, X_eval= self.X_test_orig, Y_eval= self.Y_test_orig)

    def train_test(self, p):
        self.define_model(p, p["SHUFFLE"])
        if p["BUILD"]:
            print("building model ...")
            self.model.build()
            print("build complete ...")
        print("loading model ...")
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        print("loading complete ...")
        resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "a")
        return self.run_model(p["N_EPOCH"], p, p["SHUFFLE"], X_train= self.X_train_orig, Y_train= self.Y_train_orig, Z_train= self.Z_train_orig, X_eval= self.X_test_orig, Y_eval= self.Y_test_orig, resfile= resfile)
        
