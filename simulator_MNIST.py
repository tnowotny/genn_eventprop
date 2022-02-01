import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
import mnist
from models import *
import os


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
p= {}
p["DEBUG"]= True
p["OUT_DIR"]= "."
p["TRAIN"]= True
p["DT_MS"] = 0.1
p["BUILD"] = True
p["TIMING"] = True
p["TRAIN_DATA_SEED"]= 123
p["TEST_DATA_SEED"]= 456

# Experiment parameters
p["TRIAL_MS"]= 20.0
p["N_MAX_SPIKE"]= 400    # make buffers for maximally 400 spikes (200 in a 30 ms trial) - should be safe
p["N_BATCH"]= 32
p["N_TRAIN"]= 55000
p["N_VALIDATE"]= 5000
p["N_EPOCH"]= 10
p["SHUFFLE"]= True
p["N_TEST"]= 10000
N_CLASS= 10
p["W_REPORT_INTERVAL"] = 100
p["W_EPOCH_INTERVAL"] = 10
# Network structure
NUM_INPUT = 28*28
NUM_OUTPUT = 16  # padded to power of two to be able to use warp reduction
p["NUM_HIDDEN"] = 350

# Model parameters
p["TAU_SYN"] = 5.0
p["TAU_MEM"] = 20.0
p["V_THRESH"] = 1.0
p["V_RESET"] = 0.0
p["INPUT_HIDDEN_MEAN"]= 0.078
p["INPUT_HIDDEN_STD"]= 0.045
p["HIDDEN_OUTPUT_MEAN"]= 0.2
p["HIDDEN_OUTPUT_STD"]= 0.37
p["PDROP_INPUT"] = 0.2
# Learning parameters
p["ETA"]= 5e-3
p["ADAM_BETA1"]= 0.9      
p["ADAM_BETA2"]= 0.999    
p["ADAM_EPS"]= 1e-8       
# applied every epoch
p["ETA_DECAY"]= 0.95      

# spike recording
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["REC_SPIKES"] = []
p["REC_NEURONS"] = []
p["REC_SYNAPSES"] = []
p["WRITE_TO_DISK"]= True
p["LOAD_LAST"]= False

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
        self.generate_training_data(p)
        self.generate_testing_data(p)
                
    def loss_func(self, Y, p):
        expsum= self.output.vars["expsum"].view
        max_V= self.output.vars["max_V"].view
        max_V_correct= np.array([ max_V[i,y] for i, y in enumerate(Y) ])
        loss= -np.sum(np.log(np.exp(max_V_correct)/expsum[:,0]))/p["N_BATCH"]
        return loss
    
    def generate_training_data(self, p, shuffle= True):
        X = mnist.train_images()
        Y = mnist.train_labels()
        if p["TRAIN_DATA_SEED"] is not None:
            self.datarng= np.random.default_rng(p["TRAIN_DATA_SEED"])
        else:
            self.datarng= np.random.default_rng()        
        idx= np.arange(60000)
        if (shuffle):
            self.datarng.shuffle(idx)
        X= X[idx]
        self.X_val_orig= X[60000-p["N_VALIDATE"]:,:,:]
        self.X_train_orig= X[:p["N_TRAIN"],:,:]
        Y= Y[idx]
        self.Y_val_orig= Y[60000-p["N_VALIDATE"]:]
        self.Y_train_orig= Y[:p["N_TRAIN"]]
        
    def generate_testing_data(self, p, shuffle= True):
        X = mnist.test_images()
        Y = mnist.test_labels()
        if p["TEST_DATA_SEED"] is not None:
            self.tdatarng= np.random.default_rng(p["TEST_DATA_SEED"])
        else:
            self.tdatarng= np.random.default_rng()        
        idx= np.arange(10000)
        if (shuffle):
            self.tdatarng.shuffle(idx)
        X= X[idx]
        self.X_test_orig= X[:p["N_TEST"],:,:]
        Y= Y[idx]
        self.Y_test_orig= Y[:p["N_TEST"]]
        
    def generate_input_spiketimes(self, p, Xin, Yin):
        # N is the number of training/testing images: always use all images given
        N= Xin.shape[0]
        X= Xin.copy()
        Y= Yin.copy()
        """
        if p["DEBUG"]:
            for i in range(10):
                plt.figure()
                plt.imshow(X[i*p["N_BATCH"],:,:])
                print(Y[i*p["N_BATCH"]])
        """
        X= np.reshape(X,(N, NUM_INPUT))
        # list of spike time lists
        sts= [ [] for _ in range(p["N_BATCH"]*NUM_INPUT) ]
        reps= N // p["N_BATCH"]  # number of trials run
        t_off= 0.0
        for i in range(reps):
            for j in range(p["N_BATCH"]):
                # index of the input image
                i_input= i*p["N_BATCH"]+j
                # starting neuron index of jth instance within the batch
                strt= j*NUM_INPUT
                # loop through all neurons
                for k in range(NUM_INPUT):
                    t= X[i_input,k]
                    if t > 1:   # only make a spike for gray values greater 1
                        t= (255.0-t)/255.0*(p["TRIAL_MS"]-4*p["DT_MS"])+2*p["DT_MS"]   # make sure spikes are two timesteps within the presentation window
                        sts[strt+k].append(t_off+t)
            t_off += p["TRIAL_MS"]        
        X= np.hstack(sts)
        n_sts= [ len(s) for s in sts ]
        input_end= np.cumsum(n_sts)
        input_start= np.zeros(p["N_BATCH"]*NUM_INPUT, dtype= int)
        input_start[1:]= input_end[:-1]
        input_end = np.reshape(input_end, (p["N_BATCH"], NUM_INPUT))
        input_start = np.reshape(input_start, (p["N_BATCH"], NUM_INPUT))

        #for i in range(NUM_INPUT):
        #    print(input_start[0,i])
        #    print("X: {}".format(X[input_start[0,i]:input_end[0,i]]))
        #    print("start= {}, end= {}".format(input_start[0,i],input_end[0,i]))
        return (X, Y, input_start, input_end) 

    """ 
    generate a spikeTimes array and startSpike and endSpike arrays to allow indexing into the 
    spikeTimes in a shuffled way
    """
    def generate_input_spiketimes_shuffle_fast(self, p, Xin, Yin):
        # N is the number of training/testing images: always use all images given
        N= Xin.shape[0]
        X= Xin.copy()
        Y= Yin.copy()
        """
        if p["DEBUG"]:
            for i in range(4):
                plt.figure()
                plt.imshow(X[i*p["N_BATCH"],:,:])
                print(Y[i*p["N_BATCH"]])
        """
        X= np.reshape(X,(N, NUM_INPUT))
        all_sts= []
        all_input_end= []
        all_input_start= []
        stidx_offset= 0
        for i in range(N):
            # list of spike time lists
            sts= [ [] for _ in range(NUM_INPUT) ]
            # loop through all neurons
            for k in range(NUM_INPUT):
                t= X[i,k]
                if t > 1:   # only make a spike for gray values greater 1
                    t= (255.0-t)/255.0*(p["TRIAL_MS"]-4*p["DT_MS"])+2*p["DT_MS"]   # make sure spikes are two timesteps within the presentation window
                    sts[k].append(t)
            all_sts.append(np.hstack(sts))
            n_sts= [ len(s) for s in sts ]
            i_end= np.cumsum(n_sts)+stidx_offset
            i_start= np.empty(i_end.shape)
            i_start[0]= stidx_offset
            i_start[1:]= i_end[:-1]
            all_input_end.append(i_end)
            all_input_start.append(i_start)
            stidx_offset= i_end[-1]
        X= np.hstack(all_sts)
        input_end= np.hstack(all_input_end)
        input_start= np.hstack(all_input_start)
        
        #for i in range(NUM_INPUT):
        #    print(input_start[0,i])
        #    print("X: {}".format(X[input_start[0,i]:input_end[0,i]]))
        #    print("start= {}, end= {}".format(input_start[0,i],input_end[0,i]))
        return (X, Y, input_start, input_end) 
                
    def define_model(self, p, shuffle):
        input_params= {"N_neurons": NUM_INPUT,
                       "N_max_spike": 2  # input neurons have at most one input spike per trial (is the circular spike buffer overkill??)
        }
        self.input_init_vars= {"startSpike": 0.0,  # to be set later
                          "endSpike": 0.0,         # to be set later
                          "back_spike": 0,
                          "rp_ImV": 0,
                          "wp_ImV": 0,
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
                           "rp_ImV": 0,
                           "wp_ImV": 0,
                           "back_spike": 0,
                           "lambda_jump": 0.0,
        }
        output_params= {"tau_m": p["TAU_MEM"],
                        "tau_syn": p["TAU_SYN"],
                        "trial_t": p["TRIAL_MS"],
                        "N_batch": p["N_BATCH"],
        }
        self.output_init_vars= {"V": p["V_RESET"],
                           "lambda_V": 0.0,
                           "lambda_I": 0.0,
                           "rev_t": 0.0,
                           "max_V": p["V_RESET"],
                           "new_max_V": p["V_RESET"],
                           "max_t": 0.0,
                           "new_max_t": 0.0, 
                           "expsum": 1.0,
                           "trial": 0,
                           "lambda_jump": 0.0,
        }
        # ----------------------------------------------------------------------------
        # Synapse initialisation
        # ----------------------------------------------------------------------------
        self.in_to_hid_init_vars= {"dw": 0}
        self.in_to_hid_init_vars["w"]= genn_model.init_var("Normal", {"mean": p["INPUT_HIDDEN_MEAN"], "sd": p["INPUT_HIDDEN_STD"]})

        self.hid_to_out_init_vars= {"dw": 0}
        self.hid_to_out_init_vars["w"]= genn_model.init_var("Normal", {"mean": p["HIDDEN_OUTPUT_MEAN"], "sd": p["HIDDEN_OUTPUT_STD"]})

        # ----------------------------------------------------------------------------
        # Optimiser initialisation
        # ----------------------------------------------------------------------------
        adam_params = {"beta1": p["ADAM_BETA1"], "beta2": p["ADAM_BETA2"], "epsilon": p["ADAM_EPS"], "tau_syn": p["TAU_SYN"], "N_batch": p["N_BATCH"]}
        self.adam_init_vars = {"m": 0.0, "v": 0.0}

        # ----------------------------------------------------------------------------
        # Model description
        # ----------------------------------------------------------------------------
        self.model = genn_model.GeNNModel("float", "eventprop_MNIST", generateLineInfo=True, time_precision="double")
        self.model.dT = p["DT_MS"]
        self.model.timing_enabled = p["TIMING"]
        self.model.batch_size = p["N_BATCH"]
        #model._model.set_seed(p["DATA_SEED"])

        # Add neuron populations
        if shuffle:
            self.input = self.model.add_neuron_population("input", NUM_INPUT, EVP_SSA_MNIST_SHUFFLE, 
                                                          input_params, self.input_init_vars)
        else:
            self.input = self.model.add_neuron_population("input", NUM_INPUT, EVP_SSA_MNIST, 
                                                          input_params, self.input_init_vars)
        self.input.set_extra_global_param("t_k",-1e5*np.ones(p["N_BATCH"]*NUM_INPUT*p["N_MAX_SPIKE"],dtype=np.float32))
        self.input.set_extra_global_param("spikeTimes", np.zeros(10000000,dtype=np.float32)) # reserve enough space for any set of input spikes that is likely
       
        self.hidden= self.model.add_neuron_population("hidden", p["NUM_HIDDEN"], EVP_LIF, hidden_params, self.hidden_init_vars) 
        self.hidden.set_extra_global_param("t_k",-1e5*np.ones(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"],dtype=np.float32))
        self.hidden.set_extra_global_param("ImV",np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"],dtype=np.float32))
        
        self.output= self.model.add_neuron_population("output", NUM_OUTPUT, EVP_LIF_output, output_params, self.output_init_vars)
        self.output.set_extra_global_param("t_k",-1e5*np.ones(p["N_BATCH"]*NUM_OUTPUT*p["N_MAX_SPIKE"],dtype=np.float32))
        self.output.set_extra_global_param("ImV",np.zeros(p["N_BATCH"]*NUM_OUTPUT*p["N_MAX_SPIKE"],dtype=np.float32))

        self.output.set_extra_global_param("label", np.zeros(60000,dtype=np.float32)) # reserve space for labels

        input_var_refs= {"rp_ImV": genn_model.create_var_ref(self.input, "rp_ImV"),
                         "wp_ImV": genn_model.create_var_ref(self.input, "wp_ImV"),
                         "back_spike": genn_model.create_var_ref(self.input, "back_spike"),
                         "rev_t": genn_model.create_var_ref(self.input, "rev_t")
        }
        self.input_reset= self.model.add_custom_update("input_reset","neuronReset", EVP_input_reset_MNIST, {"N_max_spike": p["N_MAX_SPIKE"]}, {}, input_var_refs)

        input_set_params= {"N_batch": p["N_BATCH"],
                           "num_input": NUM_INPUT
        }
        input_var_refs= {"startSpike": genn_model.create_var_ref(self.input, "startSpike"),
                         "endSpike": genn_model.create_var_ref(self.input, "endSpike")
                         }
        if shuffle:
            self.input_set= self.model.add_custom_update("input_set", "inputUpdate", EVP_input_set_MNIST_shuffle, input_set_params, {}, input_var_refs)
            # reserving memory for the worst case of the full training set
            self.input_set.set_extra_global_param("allStartSpike", np.zeros(60000*NUM_INPUT,dtype= int))
            self.input_set.set_extra_global_param("allEndSpike", np.zeros(60000*NUM_INPUT,dtype= int))
            self.input_set.set_extra_global_param("allInputID", np.zeros(60000,dtype= int))
            self.input_set.set_extra_global_param("trial", 0)
            
        hidden_var_refs= {"rp_ImV": genn_model.create_var_ref(self.hidden, "rp_ImV"),
                          "wp_ImV": genn_model.create_var_ref(self.hidden, "wp_ImV"),
                          "V": genn_model.create_var_ref(self.hidden, "V"),
                          "lambda_V": genn_model.create_var_ref(self.hidden, "lambda_V"),
                          "lambda_I": genn_model.create_var_ref(self.hidden, "lambda_I"),
                          "rev_t": genn_model.create_var_ref(self.hidden, "rev_t"),
                          "back_spike": genn_model.create_var_ref(self.hidden, "back_spike")
        }
        self.hidden_reset= self.model.add_custom_update("hidden_reset","neuronReset", EVP_neuron_reset, {"V_reset": p["V_RESET"], "N_max_spike": p["N_MAX_SPIKE"]}, {}, hidden_var_refs)

        output_reset_params= {"V_reset": p["V_RESET"],
                              "N_class": N_CLASS
        }
        output_var_refs= {"max_V": genn_model.create_var_ref(self.output, "max_V"),
                          "new_max_V": genn_model.create_var_ref(self.output, "new_max_V"),
                          "max_t": genn_model.create_var_ref(self.output, "max_t"),
                          "new_max_t": genn_model.create_var_ref(self.output, "new_max_t"),
                          "V": genn_model.create_var_ref(self.output, "V"),
                          "lambda_V": genn_model.create_var_ref(self.output, "lambda_V"),
                          "lambda_I": genn_model.create_var_ref(self.output, "lambda_I"),
                          "rev_t": genn_model.create_var_ref(self.output, "rev_t"),
                          "expsum": genn_model.create_var_ref(self.output, "expsum"),
                          "trial": genn_model.create_var_ref(self.output, "trial")
        }
        self.output_reset= self.model.add_custom_update("output_reset","neuronReset", EVP_neuron_reset_output_MNIST, output_reset_params, {}, output_var_refs)

        # synapse populations
        self.in_to_hid= self.model.add_synapse_population("in_to_hid", "DENSE_INDIVIDUALG", NO_DELAY, self.input, self.hidden, EVP_input_synapse,
                                                {}, self.in_to_hid_init_vars, {}, {}, my_Exp_Curr, {"tau": p["TAU_SYN"]}, {}
        )
        self.hid_to_out= self.model.add_synapse_population("hid_to_out", "DENSE_INDIVIDUALG", NO_DELAY, self.hidden, self.output, EVP_synapse,
                                                 {}, self.hid_to_out_init_vars, {}, {}, my_Exp_Curr, {"tau": p["TAU_SYN"]}, {}
        )        
        var_refs = {"dw": genn_model.create_wu_var_ref(self.in_to_hid, "dw")}
        self.in_to_hid_reduce= self.model.add_custom_update("in_to_hid_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
        var_refs = {"gradient": genn_model.create_wu_var_ref(self.in_to_hid_reduce, "reduced_dw"),
                    "variable": genn_model.create_wu_var_ref(self.in_to_hid, "w")}
        self.in_to_hid_learn= self.model.add_custom_update("in_to_hid_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)

        var_refs = {"dw": genn_model.create_wu_var_ref(self.hid_to_out, "dw")}
        self.hid_to_out_reduce= self.model.add_custom_update("hid_to_out_reduce","EVPReduce", EVP_grad_reduce, {}, {"reduced_dw": 0.0}, var_refs)
        var_refs = {"gradient": genn_model.create_wu_var_ref(self.hid_to_out_reduce, "reduced_dw"),
                    "variable": genn_model.create_wu_var_ref(self.hid_to_out, "w")}
        self.hid_to_out_learn= self.model.add_custom_update("hid_to_out_learn","EVPLearn", adam_optimizer_model, adam_params, self.adam_init_vars, var_refs)
        self.hid_to_out.pre_target_var= "revIsyn"

        self.optimisers= [self.in_to_hid_learn, self.hid_to_out_learn]
        #self.optimisers= [self.hid_to_out_learn]
        # global normalisation of hid to out synapses
        var_refs = {"w": genn_model.create_wu_var_ref(self.hid_to_out, "w")}
        self.normalize_out=  self.model.add_custom_update("normalize_hid_to_out", "Normalize", normalize_model, {}, {}, var_refs)
        
        # enable buffered spike recording where desired
        for pop in p["REC_SPIKES"]:
            self.model.neuron_populations[pop].spike_recording_enabled= True

    """
    ----------------------------------------------------------------------------
    Run the model
    ----------------------------------------------------------------------------
    """
            
    def run_model(self, number_epochs, learning, labels, X_t_orig, N_trial, p, shuffle, do_shuffle= False):
        if p["LOAD_LAST"]:
            self.in_to_hid.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], "w_input_hidden_last.npy"))
            self.hid_to_out.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], "w_hidden_output_last.npy"))
            self.in_to_hid.push_var_to_device("w")
            self.hid_to_out.push_var_to_device("w")
        else:
            # zero the weights of synapses to "padding output neurons" - this should remove them
            # from influencing the backward pass
            mask= np.zeros((p["NUM_HIDDEN"],NUM_OUTPUT))
            mask[:,N_CLASS:]= 1
            mask= np.array(mask, dtype= bool).flatten()
            self.hid_to_out.pull_var_from_device("w")
            self.hid_to_out.vars["w"].view[mask]= 0
            self.hid_to_out.push_var_to_device("w")
        self.hid_to_out.pull_var_from_device("w")
        plt.figure()
        plt.hist(self.hid_to_out.vars["w"].view[:],100)
        adam_step= 1
        learning_rate= p["ETA"]
        cnt= np.ones(p["N_BATCH"])
        spike_t= {}
        spike_ID= {}
        for pop in p["REC_SPIKES"]:
            spike_t[pop]= []
            spike_ID[pop]= []

        rec_vars_n= {}
        for pop, var in p["REC_NEURONS"]:
            rec_vars_n[var+pop]= []
        rec_vars_s= {}
        for pop, var in p["REC_SYNAPSES"]:
            rec_vars_s[var+pop]= []
        # build and assign the input spike train and corresponding labels
        if shuffle:
            X, Y, input_start, input_end= self.generate_input_spiketimes_shuffle_fast(p, X_t_orig, labels)
            self.input.extra_global_params["spikeTimes"].view[:len(X)]= X
            self.input.push_extra_global_param_to_device("spikeTimes")
            self.input_set.extra_global_params["allStartSpike"].view[:len(input_start)]= input_start
            self.input_set.push_extra_global_param_to_device("allStartSpike")
            self.input_set.extra_global_params["allEndSpike"].view[:len(input_end)]= input_end
            self.input_set.push_extra_global_param_to_device("allEndSpike")
            input_id= np.arange(X_t_orig.shape[0])
        else:
            X, Y, input_start, input_end= self.generate_input_spiketimes(p, X_t_orig, labels)
            self.input.extra_global_params["spikeTimes"].view[:len(X)]= X
            self.input.push_extra_global_param_to_device("spikeTimes")
            self.output.extra_global_params["label"].view[:len(Y)]= Y
            self.output.push_extra_global_param_to_device("label")
            self.input_init_vars["startSpike"]= input_start
            self.input_init_vars["endSpike"]= input_end
        first= True
        for epoch in range(number_epochs):
            if shuffle:
                if do_shuffle:
                    self.datarng.shuffle(input_id)
                #print(input_id)
                Y= labels[input_id]
                self.output.extra_global_params["label"].view[:len(Y)]= Y
                self.output.push_extra_global_param_to_device("label")
                self.input_set.extra_global_params["allInputID"].view[:len(input_id)]= input_id
                self.input_set.push_extra_global_param_to_device("allInputID")
            predict= []
            the_loss= []
            good= 0.0    
            self.model.t= 0.0
            self.model.timestep= 0
            for var, val in self.input_init_vars.items():
                self.input.vars[var].view[:]= val
            self.input.push_state_to_device()
            for var, val in self.hidden_init_vars.items():
                self.hidden.vars[var].view[:]= val
            self.hidden.push_state_to_device()
            for var, val in self.output_init_vars.items():
                self.output.vars[var].view[:]= val
            self.output.push_state_to_device()
            for trial in range(N_trial):
                trial_end= (trial+1)*p["TRIAL_MS"]
                # if shuffling: assign the input spike train and corresponding labels
                if shuffle:
                    self.input_set.extra_global_params["trial"].view[:]= trial
                    self.model.custom_update("inputUpdate")
                    self.input.extra_global_params["t_offset"].view[:]= self.model.t
                    
                int_t= 0
                while (self.model.t < trial_end-1e-1*p["DT_MS"]):
                    self.model.step_time()
                    int_t += 1
                    if len(p["REC_SPIKES"]) > 0:
                        if int_t%p["SPK_REC_STEPS"] == 0:
                            self.model.pull_recording_buffers_from_device()
                            for pop in p["REC_SPIKES"]:
                                the_pop= self.model.neuron_populations[pop]
                                if p["N_BATCH"] > 1:
                                    spike_t[pop].append(the_pop.spike_recording_data[0][0]+epoch*N_trial*p["TRIAL_MS"])
                                    spike_ID[pop].append(the_pop.spike_recording_data[0][1])
                                else:
                                    spike_t[pop].append(the_pop.spike_recording_data[0]+epoch*N_trial*p["TRIAL_MS"])
                                    spike_ID[pop].append(the_pop.spike_recording_data[1])

                    for pop, var in p["REC_NEURONS"]:
                        the_pop= self.model.neuron_populations[pop]
                        the_pop.pull_var_from_device(var)
                        rec_vars_n[var+pop].append(the_pop.vars[var].view[0].copy())

                    for pop, var in p["REC_SYNAPSES"]:
                        the_pop= self.model.synapse_populations[pop]
                        if var == "in_syn":
                            the_pop.pull_in_syn_from_device()
                            rec_vars_s[var+pop].append(the_pop.in_syn[0].copy())
                        else:
                            the_pop.pull_var_from_device(var)
                            rec_vars_s[var+pop].append(the_pop.vars[var].view.copy())
                    if np.abs(self.model.t + p["DT_MS"] - trial_end) < 1e-1*p["DT_MS"]:
                        self.in_to_hid.in_syn[:]= 0.0
                        self.in_to_hid.push_in_syn_to_device()
                        self.hid_to_out.in_syn[:]= 0.0
                        self.hid_to_out.push_in_syn_to_device()
                
                if learning:
                    update_adam(learning_rate, adam_step, self.optimisers)
                    adam_step += 1
                    self.model.custom_update("EVPReduce")
                    #if trial%2 == 1:
                    self.model.custom_update("EVPLearn")
                    #self.normalize_out.extra_global_params["expsum"].view[:]= np.mean(self.output.vars["expsum"].view[:][0])
                    #self.model.custom_update("Normalize")
                self.in_to_hid.in_syn[:]= 0.0
                self.in_to_hid.push_in_syn_to_device()
                self.hid_to_out.in_syn[:]= 0.0
                self.hid_to_out.push_in_syn_to_device()
                self.model.custom_update("neuronReset")
                # record training loss and error
                # NOTE: the neuronReset does the calculation of expsum and updates max_V
                # from new_max_V, so use max_V here!
                self.output.pull_var_from_device("max_V")
                #print(self.output.vars["max_V"].view)
                #print(self.output.vars["max_V"].view.shape)
                pred= np.argmax(self.output.vars["max_V"].view, axis=-1)
                #print(pred)
                lbl= Y[trial*p["N_BATCH"]:(trial+1)*p["N_BATCH"]]
                #print(lbl)
                #print("-----")
                good += np.sum(cnt[pred == lbl])
                predict.append(pred)
                self.output.pull_var_from_device("expsum")
                losses= self.loss_func(lbl,p)   # uses self.output.vars["max_V"].view and self.output.vars["expsum"].view
                the_loss.append(losses)

                if (epoch % p["W_EPOCH_INTERVAL"] == 0) and (trial % p["W_REPORT_INTERVAL"] == 0):
                    self.in_to_hid.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], "w_input_hidden_e{}_t{}.npy".format(epoch,trial)), self.in_to_hid.vars["w"].view.copy())
                    self.hid_to_out.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], "w_hidden_output_e{}_t{}.npy".format(epoch,trial)), self.hid_to_out.vars["w"].view.copy())

            #print(the_loss)
            print("{} Correct: {}, Loss: {}".format(epoch, good/(N_trial*p["N_BATCH"]),np.mean(the_loss)))
            predict= np.hstack(predict)
            if learning:
                learning_rate *= p["ETA_DECAY"]
            first= False
                
        for pop in p["REC_SPIKES"]:
            spike_t[pop]= np.hstack(spike_t[pop])
            spike_ID[pop]= np.hstack(spike_ID[pop])

        for pop, var in p["REC_NEURONS"]:
            rec_vars_n[var+pop]= np.vstack(rec_vars_n[var+pop])
        
        for pop, var in p["REC_SYNAPSES"]:
            rec_vars_s[var+pop]= np.vstack(rec_vars_s[var+pop])
        
        if p["WRITE_TO_DISK"]:            # Saving results
            for pop in p["REC_SPIKES"]:
                np.save(p["OUT_DIR"]+"/"+pop+"_spike_t", spike_t[pop])
                np.save(p["OUT_DIR"]+"/"+pop+"_spike_ID", spike_ID[pop])

            for pop, var in p["REC_NEURONS"]:
                np.save(p["OUT_DIR"]+"/"+var+pop, rec_vars_n[var+pop])

            for pop, var in p["REC_SYNAPSES"]:
                np.save(p["OUT_DIR"]+"/"+var+pop, rec_vars_s[var+pop])

        self.in_to_hid.pull_var_from_device("w")
        self.hid_to_out.pull_var_from_device("w")
        np.save(os.path.join(p["OUT_DIR"], "w_input_hidden_last.npy"), self.in_to_hid.vars["w"].view.copy())
        np.save(os.path.join(p["OUT_DIR"], "w_hidden_output_last.npy"), self.hid_to_out.vars["w"].view.copy())
        return (spike_t, spike_ID, rec_vars_n, rec_vars_s, good/(N_trial*p["N_BATCH"]))
        
    def train(self, p):
        self.define_model(p, p["SHUFFLE"])
        if p["BUILD"]:
            self.model.build()
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        N_trial= p["N_TRAIN"] // p["N_BATCH"]
        self.input.extra_global_params["pDrop"].view[:]= p["PDROP_INPUT"]    # set dropout
        return self.run_model(p["N_EPOCH"], True, self.Y_train_orig, self.X_train_orig, N_trial, p, p["SHUFFLE"],do_shuffle= p["SHUFFLE"])
          
    def test(self, p):
        self.define_model(p, False)
        if p["BUILD"]:
            self.model.build()
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        N_trial= p["N_TEST"] // p["N_BATCH"]
        self.input.extra_global_params["pDrop"].view[:]= 0.0          # no dropout during testing
        return self.run_model(1, False, self.Y_test_orig, self.X_test_orig, N_trial, p, False)
        
