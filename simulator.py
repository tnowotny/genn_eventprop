import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from Dataset import YinYangDataset
from models import *
import os

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
p= {}
p["OUT_DIR"]= "."
p["TRAIN"]= True
p["DT_MS"] = 0.1
p["BUILD"] = True
p["TIMING"] = True
p["TRAIN_DATA_SEED"]= 123
p["TEST_DATA_SEED"]= 456

# Experiment parameters
p["TRIAL_MS"]= 30.0
p["N_MAX_SPIKE"]= 400    # make buffers for maximally 60 spikes (30 in a 30 ms trial) - should be safe
p["N_BATCH"]= 32
p["N_TRAIN"]= p["N_BATCH"]*1000
p["N_EPOCH"]= 10
p["N_TEST"]= p["N_BATCH"]*25
N_CLASS= 3
p["W_REPORT_INTERVAL"] = 100
p["W_EPOCH_INTERVAL"] = 10
# Network structure
NUM_INPUT = 5
NUM_OUTPUT = N_CLASS
p["NUM_HIDDEN"] = 200

# Model parameters
p["TAU_SYN"] = 5.0
p["TAU_MEM"] = 20.0
p["V_THRESH"] = 1.0
p["V_RESET"] = 0.0
p["TAU_0"]= 0.5
p["TAU_1"]= 6.4
p["ALPHA"]= 3e-3
p["INPUT_HIDDEN_MEAN"]= 1.5
p["INPUT_HIDDEN_STD"]= 0.78
p["HIDDEN_OUTPUT_MEAN"]= 0.93
p["HIDDEN_OUTPUT_STD"]= 0.1

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
p["TRAINING_PLOT"]= False
p["TRAINING_PLOT_INTERVAL"]= 10
p["FANCY_PLOTS"]= False
p["LOAD_LAST"]= False

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
    first_moment_scale = 1.0 / (1.0 - (p["ADAM_BETA1"] ** adam_step))
    second_moment_scale = 1.0 / (1.0 - (p["ADAM_BETA2"] ** adam_step))
    #print(first_moment_scale)
    #print(second_moment_scale)
    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["alpha"].view[:] = learning_rate
        o.extra_global_params["firstMomentScale"].view[:] = first_moment_scale
        o.extra_global_params["secondMomentScale"].view[:] = second_moment_scale

def loss_func(nfst, Y, trial):
    #print("new first spikes: {}".format(nfst))
    t= nfst-trial*p["TRIAL_MS"]
    t[t < 0.0]= p["TRIAL_MS"]
    expsum= np.sum(np.exp(-t/p["TAU_0"]),axis=-1)
    pred= np.argmin(t,axis=-1)
    selected= np.array([ t[i,pred[i]] for i in range(pred.shape[0])])
    #print("expsum: {}, pred: {}, selected: {}".format(expsum,pred,selected))
    #loss= -np.sum(np.log(np.exp(-selected/p["TAU_0"])/expsum)-p["ALPHA"]*(np.exp(selected/p["TAU_1"])-1))
    loss= -np.sum(np.log(np.exp(-selected/p["TAU_0"])/expsum)-p["ALPHA"]/(1.01*p["TRIAL_MS"]-selected))
    #loss= -np.sum(np.log(np.exp(-selected/p["TAU_0"])/expsum))
    #print(np.sum(p["ALPHA"]/(1.05*p["TRIAL_MS"]-selected)))
    loss/= p["N_BATCH"]
    return loss

class yingyang:
    def __init__(self, p):
        self.generate_training_data(p)
        self.generate_testing_data(p)

    def generate_training_data(self, p):
        if p["TRAIN_DATA_SEED"] is not None:
            np.random.seed(p["TRAIN_DATA_SEED"])
        X, self.Y_train = YinYangDataset(size=p["N_TRAIN"]*N_CLASS, flipped_coords=True, seed=p["TRAIN_DATA_SEED"])[:] 
        self.X_train_orig= X[:,0:2]
        np.save("training_labels.npy", self.Y_train)
        X= X.T
        z= np.zeros(p["N_TRAIN"] * N_CLASS)
        X= np.vstack([z, X])
        X= np.vstack([ X[:,i:p["N_TRAIN"]*N_CLASS:p["N_BATCH"]] for i in range(p["N_BATCH"])])
        chunk= (p["N_TRAIN"] * N_CLASS) // p["N_BATCH"]
        offset= np.reshape(np.arange(0,chunk * p["TRIAL_MS"], p["TRIAL_MS"]),(1,chunk))
        offset= np.repeat(offset,NUM_INPUT*p["N_BATCH"],axis=0)
        X= X*(p["TRIAL_MS"]-4*p["DT_MS"])+offset+2*p["DT_MS"]
        self.X_train= X.flatten()
        self.input_end_train = np.arange(chunk,NUM_INPUT*p["N_BATCH"]*chunk+1, chunk)
        self.input_end_train = np.reshape(self.input_end_train, (p["N_BATCH"], NUM_INPUT))
        self.input_start_train = get_input_start(self.input_end_train)

    def generate_testing_data(self, p):
        if p["TEST_DATA_SEED"] is not None:
            np.random.seed(p["TEST_DATA_SEED"])
        X, self.Y_test = YinYangDataset(size=p["N_TEST"]*N_CLASS, flipped_coords=True, seed=p["TEST_DATA_SEED"])[:]
        self.X_test_orig= X[:,0:2]
        X= X.T
        z= np.zeros(p["N_TEST"] * N_CLASS)
        X= np.vstack([z, X])
        X= np.vstack([ X[:,i:p["N_TEST"]*N_CLASS:p["N_BATCH"]] for i in range(p["N_BATCH"])])
        chunk= (p["N_TEST"] * N_CLASS) // p["N_BATCH"]
        offset= np.reshape(np.arange(0,chunk * p["TRIAL_MS"], p["TRIAL_MS"]),(1,chunk))
        offset= np.repeat(offset,NUM_INPUT*p["N_BATCH"],axis=0)
        X= X*(p["TRIAL_MS"]-4*p["DT_MS"])+offset+2*p["DT_MS"]
        self.X_test= X.flatten()
        self.input_end_test = np.arange(chunk,NUM_INPUT*p["N_BATCH"]*chunk+1, chunk)
        self.input_end_test = np.reshape(self.input_end_test, (p["N_BATCH"], NUM_INPUT))
        self.input_start_test = get_input_start(self.input_end_test)
        
    def define_model(self, input_spike_times, output_labels, p):
        input_params= {}
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
                        "V_thresh": p["V_THRESH"],
                        "V_reset": p["V_RESET"],
                        "N_neurons": NUM_OUTPUT,
                        "N_max_spike": p["N_MAX_SPIKE"],
                        "tau_syn": p["TAU_SYN"],
                        "trial_t": p["TRIAL_MS"],
                        "tau0": p["TAU_0"],
                        "tau1": p["TAU_1"],
                        "alpha": p["ALPHA"],
                        "N_batch": p["N_BATCH"],
        }
        self.output_init_vars= {"V": p["V_RESET"],
                           "lambda_V": 0.0,
                           "lambda_I": 0.0,
                           "rev_t": 0.0,
                           "rp_ImV": 0,
                           "wp_ImV": 0,
                           "back_spike": 0,
                           "first_spike_t": -1e5, 
                           "new_first_spike_t": -1e5,
                           "expsum": 1.0,
                           "trial": 0,
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
        self.model = genn_model.GeNNModel("float", "eventprop_yingyang", generateLineInfo=True, time_precision="double")
        self.model.dT = p["DT_MS"]
        self.model.timing_enabled = p["TIMING"]
        self.model.batch_size = p["N_BATCH"]
        #model._model.set_seed(p["DATA_SEED"])

        # Add neuron populations
        self.input = self.model.add_neuron_population("input", NUM_INPUT, EVP_SSA, 
                                            {}, self.input_init_vars)
        
        self.hidden= self.model.add_neuron_population("hidden", p["NUM_HIDDEN"], EVP_LIF, hidden_params, self.hidden_init_vars) 
        self.hidden.set_extra_global_param("t_k",-1e5*np.ones(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"],dtype=np.float32))
        self.hidden.set_extra_global_param("ImV",np.zeros(p["N_BATCH"]*p["NUM_HIDDEN"]*p["N_MAX_SPIKE"],dtype=np.float32))
        
        self.output= self.model.add_neuron_population("output", NUM_OUTPUT, EVP_LIF_output, output_params, self.output_init_vars)
        self.output.set_extra_global_param("t_k",-1e5*np.ones(p["N_BATCH"]*NUM_OUTPUT*p["N_MAX_SPIKE"],dtype=np.float32))
        self.output.set_extra_global_param("ImV",np.zeros(p["N_BATCH"]*NUM_OUTPUT*p["N_MAX_SPIKE"],dtype=np.float32))

        self.input.set_extra_global_param("spikeTimes", input_spike_times)
        self.output.set_extra_global_param("label", output_labels)

        input_var_refs= {"startSpike": genn_model.create_var_ref(self.input, "startSpike"),
                         "last_startSpike": genn_model.create_var_ref(self.input, "last_startSpike"),
                         "back_startSpike": genn_model.create_var_ref(self.input, "back_startSpike"),
                         "back_endSpike": genn_model.create_var_ref(self.input, "back_endSpike"),
                         "back_spike": genn_model.create_var_ref(self.input, "back_spike"),
                         "rev_t": genn_model.create_var_ref(self.input, "rev_t")
        }

        self.input_reset= self.model.add_custom_update("input_reset","neuronReset", EVP_input_reset, {}, {}, input_var_refs)
    
        hidden_var_refs= {"rp_ImV": genn_model.create_var_ref(self.hidden, "rp_ImV"),
                          "wp_ImV": genn_model.create_var_ref(self.hidden, "wp_ImV"),
                          "V": genn_model.create_var_ref(self.hidden, "V"),
                          "lambda_V": genn_model.create_var_ref(self.hidden, "lambda_V"),
                          "lambda_I": genn_model.create_var_ref(self.hidden, "lambda_I"),
                          "rev_t": genn_model.create_var_ref(self.hidden, "rev_t"),
                          "fwd_start": genn_model.create_var_ref(self.hidden, "fwd_start"),
                          "new_fwd_start": genn_model.create_var_ref(self.hidden, "new_fwd_start"),
                          "back_spike": genn_model.create_var_ref(self.hidden, "back_spike"),
        }
        self.hidden_reset= self.model.add_custom_update("hidden_reset","neuronReset", EVP_neuron_reset, {"V_reset": p["V_RESET"], "N_max_spike": p["N_MAX_SPIKE"]}, {}, hidden_var_refs)

        output_reset_params= {"V_reset": p["V_RESET"],
                              "N_max_spike": p["N_MAX_SPIKE"],
                              "tau0": p["TAU_0"],
                              "tau1": p["TAU_1"]
        }
        output_var_refs= {"rp_ImV": genn_model.create_var_ref(self.output, "rp_ImV"),
                          "wp_ImV": genn_model.create_var_ref(self.output, "wp_ImV"),
                          "V": genn_model.create_var_ref(self.output, "V"),
                          "lambda_V": genn_model.create_var_ref(self.output, "lambda_V"),
                          "lambda_I": genn_model.create_var_ref(self.output, "lambda_I"),
                          "rev_t": genn_model.create_var_ref(self.output, "rev_t"),
                          "back_spike": genn_model.create_var_ref(self.output, "back_spike"),
                          "first_spike_t": genn_model.create_var_ref(self.output, "first_spike_t"),
                          "new_first_spike_t": genn_model.create_var_ref(self.output, "new_first_spike_t"),
                          "expsum": genn_model.create_var_ref(self.output, "expsum"),
                          "trial": genn_model.create_var_ref(self.output, "trial")                      
        }
        self.output_reset= self.model.add_custom_update("output_reset","neuronReset", EVP_neuron_reset_output, output_reset_params, {}, output_var_refs)

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
        # enable buffered spike recording where desired
        for pop in p["REC_SPIKES"]:
            self.model.neuron_populations[pop].spike_recording_enabled= True

    """
    ----------------------------------------------------------------------------
    Run the model
    ----------------------------------------------------------------------------
    """
            
    def run_model(self, number_epochs, learning, labels, X_t_orig, N_trial, p):
        if p["LOAD_LAST"]:
            self.in_to_hid.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], "w_input_hidden_last.npy"))
            self.hid_to_out.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], "w_hidden_output_last.npy"))
            self.in_to_hid.push_var_to_device("w")
            self.hid_to_out.push_var_to_device("w")
        nfst= self.output.vars["new_first_spike_t"].view
        adam_step= 1
        learning_rate= p["ETA"]
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
        all_nfst= []
        for epoch in range(number_epochs):
            if learning:
                learning_rate *= p["ETA_DECAY"]
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
                int_t= 0
                while (self.model.t < trial_end-1e-3*p["DT_MS"]):
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
                    # clamp in_syn to 0 one timestep before trial end to avoid bleeding spikes into the next trial
                    if np.abs(self.model.t + p["DT_MS"] - trial_end) < 1e-1*p["DT_MS"]:
                        self.in_to_hid.in_syn[:]= 0.0
                        self.in_to_hid.push_in_syn_to_device()
                        self.hid_to_out.in_syn[:]= 0.0
                        self.hid_to_out.push_in_syn_to_device()
                self.output.pull_var_from_device("new_first_spike_t");
                all_nfst.append(nfst.copy())
                st= nfst.copy()
                valid= np.max(st, axis= -1) >= 0.0
                st[nfst < 0.0]= self.model.t+p["TRIAL_MS"]  # neurons that did not spike set to spike time in the future
                pred= np.argmin(st,axis=-1)
                good += np.sum(pred[valid] == labels[trial*p["N_BATCH"]:(trial+1)*p["N_BATCH"]][valid])
                predict.append(pred)
                the_loss.append(loss_func(nfst,labels[trial*p["N_BATCH"]:(trial+1)*p["N_BATCH"]],trial))
                if learning:
                    # record training loss and error
                    update_adam(learning_rate, adam_step, self.optimisers)
                    adam_step += 1
                    self.model.custom_update("EVPReduce")
                    #if trial%2 == 1:
                    self.model.custom_update("EVPLearn")                
            
                self.in_to_hid.in_syn[:]= 0.0
                self.in_to_hid.push_in_syn_to_device()
                self.hid_to_out.in_syn[:]= 0.0
                self.hid_to_out.push_in_syn_to_device()
                self.model.custom_update("neuronReset")

                if (epoch % p["W_EPOCH_INTERVAL"] == 0) and (trial % p["W_REPORT_INTERVAL"] == 0):
                    self.in_to_hid.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], "w_input_hidden_e{}_t{}.npy".format(epoch,trial)), self.in_to_hid.vars["w"].view.copy())
                    self.hid_to_out.pull_var_from_device("w")
                    np.save(os.path.join(p["OUT_DIR"], "w_hidden_output_e{}_t{}.npy".format(epoch,trial)), self.hid_to_out.vars["w"].view.copy())

            print("{} Correct: {}, Loss: {}".format(epoch, good/(N_trial*p["N_BATCH"]),np.mean(the_loss)))
            predict= np.hstack(predict)
            if p["TRAINING_PLOT"] and epoch%p["TRAINING_PLOT_INTERVAL"] == 0:
                plt.figure()
                plt.scatter(X_t_orig[:,0],X_t_orig[:,1],c=predict,s=2)
                plt.show()

            if p["FANCY_PLOTS"]:
                pltnfst= np.vstack(all_nfst)
                for i in range(3):
                    plt.figure()
                    plt.set_cmap('hot')
                    plt.scatter(X_t_orig[np.logical_and(predict == i, pltnfst[:,i] > 0.0),0],X_t_orig[np.logical_and(predict == i,pltnfst[:,i] > 0.0),1],s=30, color=[ 1, 1, 0.7],edgecolors= "black")
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.scatter(X_t_orig[pltnfst[:,i] < 0.0,0],X_t_orig[pltnfst[:,i] < 0.0,1],s=30, c='g',marker='x')
                    plt.scatter(X_t_orig[np.logical_and(predict != i, pltnfst[:,i] > 0.0),0],X_t_orig[np.logical_and(predict != i, pltnfst[:,i] > 0.0),1],s=30, c=pltnfst[np.logical_and(predict != i, pltnfst[:,i] > 0.0),i],marker='x')
                    plt.colorbar()
                    plt.clim(0, 180)
                plt.show()
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

        np.save("all_nfst.npy",np.array(all_nfst))
        self.in_to_hid.pull_var_from_device("w")
        self.hid_to_out.pull_var_from_device("w")
        np.save(os.path.join(p["OUT_DIR"], "w_input_hidden_last.npy"), self.in_to_hid.vars["w"].view.copy())
        np.save(os.path.join(p["OUT_DIR"], "w_hidden_output_last.npy"), self.hid_to_out.vars["w"].view.copy())
        return (spike_t, spike_ID, rec_vars_n, rec_vars_s, good/(N_trial*p["N_BATCH"]))
        
    def train(self, p):
        self.input_init_vars= {"startSpike": self.input_start_train,
                          "endSpike": self.input_end_train,
                          "last_startSpike": self.input_start_train,
                          "back_startSpike": self.input_start_train,
                          "back_endSpike":self. input_start_train,
                          "back_spike": 0,
                          "rev_t": 0.0}
        self.define_model(self.X_train, self.Y_train, p)
        if p["BUILD"]:
            self.model.build()
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        N_trial= (p["N_TRAIN"] * N_CLASS) // p["N_BATCH"]
        return self.run_model(p["N_EPOCH"], True, self.Y_train, self.X_train_orig, N_trial, p)
          
    def test(self, p):
        self.input_init_vars= {"startSpike": self.input_start_test,
                          "endSpike": self.input_end_test,
                          "last_startSpike": self.input_start_test,
                          "back_startSpike": self.input_start_test,
                          "back_endSpike": self.input_start_test,
                          "back_spike": 0,
                          "rev_t": 0.0}
        self.define_model(self.X_test, self.Y_test, p)
        if p["BUILD"]:
            self.model.build()
        self.model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
        N_trial= (p["N_TEST"] * N_CLASS) // p["N_BATCH"]
        return self.run_model(1, False, self.Y_test, self.X_test_orig, N_trial, p)
        
