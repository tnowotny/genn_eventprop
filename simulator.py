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
p["DATA_SEED"]= 123

# Experiment parameters
p["TRIAL_MS"]= 30.0
p["N_MAX_SPIKE"]= 200    # make buffers for maximally 60 spikes (30 in a 30 ms trial) - should be safe
p["N_BATCH"]= 32
p["N_TRAIN"]= p["N_BATCH"]*1000
p["N_EPOCH"]= 10
p["N_TEST"]= p["N_BATCH"]*100
N_CLASS= 3
p["W_REPORT_INTERVAL"] = 100

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
p["ADAM_EPS"]= 1e-8       # UNUSED - what is its intended use?
# applied every epoch
p["ETA_DECAY"]= 0.95      # UNUSED - presumably a decay in learning rate

# spike recording
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["REC_SPIKES"] = []
p["REC_NEURONS"] = []
p["REC_SYNAPSES"] = []
p["WRITE_TO_DISK"]= True

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

    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["alpha"].view[:] = learning_rate
        o.extra_global_params["firstMomentScale"].view[:] = first_moment_scale
        o.extra_global_params["secondMomentScale"].view[:] = second_moment_scale


def run_yingyang(p):
    X= []
    Y= []
    LB= []
    # Convert parameters to timesteps
    TRIAL_TIMESTEPS = int(p["TRIAL_MS"] / p["DT_MS"])

    # ----------------------------------------------------------------------------
    # Input and output preparation
    # ----------------------------------------------------------------------------

    np.random.seed(p["DATA_SEED"])
    X_train, Y_train = YinYangDataset(size=p["N_TRAIN"] * N_CLASS, 
                                      flipped_coords=True, seed=p["DATA_SEED"])[:]
    X_train= X_train.T
    z= np.zeros(p["N_TRAIN"] * N_CLASS)
    X_train= np.vstack([z, X_train])
    X_train= np.vstack([ X_train[:,i:p["N_TRAIN"] * N_CLASS:p["N_BATCH"]] for i in range(p["N_BATCH"])])
    chunk= (p["N_TRAIN"] * N_CLASS) // p["N_BATCH"]
    offset= np.reshape(np.arange(0,chunk * p["TRIAL_MS"], p["TRIAL_MS"]),(1,chunk))
    offset= np.repeat(offset,NUM_INPUT*p["N_BATCH"],axis=0)
    X_train= X_train*p["TRIAL_MS"]+offset
    X_train= X_train.flatten()
    #print(X_train)
    #print(Y_train)
    input_end_train = np.arange(chunk,NUM_INPUT*p["N_BATCH"]*chunk+1, chunk)
    input_end_train = np.reshape(input_end_train, (p["N_BATCH"], NUM_INPUT))
    input_start_train = get_input_start(input_end_train)
    
    X_test, Y_test = YinYangDataset(size=p["N_TEST"] * N_CLASS, 
                                    flipped_coords=True, seed=None)[:]
    X_t_orig= X_test[:,0:2]
    X_test= X_test.T
    z= np.zeros(p["N_TEST"] * N_CLASS)
    X_test= np.vstack([z, X_test])
    X_test= np.vstack([ X_test[:,i:p["N_TEST"]*N_CLASS:p["N_BATCH"]] for i in range(p["N_BATCH"])])
    chunk= (p["N_TEST"] * N_CLASS) // p["N_BATCH"]
    offset= np.reshape(np.arange(0,chunk * p["TRIAL_MS"], p["TRIAL_MS"]),(1,chunk))
    offset= np.repeat(offset,NUM_INPUT*p["N_BATCH"],axis=0)
    X_test= X_test*p["TRIAL_MS"]+offset
    X_test= X_test.flatten()
    
    input_end_test = np.arange(chunk,NUM_INPUT*p["N_BATCH"]*chunk+1, chunk)
    input_end_test = np.reshape(input_end_test, (p["N_BATCH"], NUM_INPUT))
    input_start_test = get_input_start(input_end_test)


    # ----------------------------------------------------------------------------
    # Neuron initialisation
    # ----------------------------------------------------------------------------

    input_params= {}
    if p["TRAIN"]:
        input_init_vars= {"startSpike": input_start_train,
                          "endSpike": input_end_train,
                          "last_startSpike": input_start_train,
                          "back_startSpike": input_start_train,
                          "back_endSpike": input_start_train,
                          "back_spike": 0,
                          "rev_t": 0.0}
    else:
        input_init_vars= {"startSpike": input_start_test,
                          "endSpike": input_end_test,
                          "last_startSpike": input_start_test,
                          "back_startSpike": input_start_test,
                          "back_endSpike": input_start_test,
                          "back_spike": 0,
                          "rev_t": 0.0}

    hidden_params= {"tau_m": p["TAU_MEM"],
                    "V_thresh": p["V_THRESH"],
                    "V_reset": p["V_RESET"],
                    "N_max_spike": p["N_MAX_SPIKE"],
                    "tau_syn": p["TAU_SYN"],
    }
    hidden_init_vars= {"V": p["V_RESET"],
                       "lambda_V": 0.0,
                       "lambda_I": 0.0,
                       "rev_t": 0.0,
                       "rp_ImV": 0,
                       "wp_ImV": 0,
                       "back_spike": 0,
    }

    output_params= {"tau_m": p["TAU_MEM"],
                    "V_thresh": p["V_THRESH"],
                    "V_reset": p["V_RESET"],
                    "N_max_spike": p["N_MAX_SPIKE"],
                    "tau_syn": p["TAU_SYN"],
                    "trial_t": p["TRIAL_MS"],
                    "tau0": p["TAU_0"],
                    "tau1": p["TAU_1"],
                    "alpha": p["ALPHA"],
                    "N_batch": p["N_BATCH"],
    }

    output_init_vars= {"V": p["V_RESET"],
                       "lambda_V": 0.0,
                       "lambda_I": 0.0,
                       "rev_t": 0.0,
                       "rp_ImV": 0,
                       "wp_ImV": 0,
                       "back_spike": 0,
                       "first_spike_t": -1e5,
                       "new_first_spike_t": -1e5,
                       "expsum": 1.0,
                       "trial": 0
    }

    # ----------------------------------------------------------------------------
    # Synapse initialisation
    # ----------------------------------------------------------------------------
    
    in_to_hid_init_vars= {"dw": 0}
    in_to_hid_init_vars["w"]= genn_model.init_var("Normal", {"mean": p["INPUT_HIDDEN_MEAN"], "sd": p["INPUT_HIDDEN_STD"]})

    hid_to_out_init_vars= {"dw": 0}
    hid_to_out_init_vars["w"]= genn_model.init_var("Normal", {"mean": p["HIDDEN_OUTPUT_MEAN"], "sd": p["HIDDEN_OUTPUT_STD"]})

    # ----------------------------------------------------------------------------
    # Optimiser initialisation
    # ----------------------------------------------------------------------------

    adam_params = {"beta1": p["ADAM_BETA1"], "beta2": p["ADAM_BETA2"], "epsilon": 1E-8, "tau_syn": p["TAU_SYN"], "N_batch": p["N_BATCH"]}
    adam_init_vars = {"m": 0.0, "v": 0.0}

    # ----------------------------------------------------------------------------
    # Model description
    # ----------------------------------------------------------------------------

    model = genn_model.GeNNModel("float", "eventprop_yingyang", generateLineInfo=True)
    model.dT = p["DT_MS"]
    model.timing_enabled = p["TIMING"]
    model.batch_size = p["N_BATCH"]

    # Add neuron populations
    input = model.add_neuron_population("input", NUM_INPUT, EVP_SSA, 
                                        {}, input_init_vars)
                                    
    hidden= model.add_neuron_population("hidden", p["NUM_HIDDEN"], EVP_LIF, hidden_params, hidden_init_vars) 
    hidden.set_extra_global_param("t_k",-1e5*np.ones(p["NUM_HIDDEN"]*p["N_MAX_SPIKE"],dtype=np.float32))
    hidden.set_extra_global_param("ImV",np.zeros(p["NUM_HIDDEN"]*p["N_MAX_SPIKE"],dtype=np.float32))

    output= model.add_neuron_population("output", NUM_OUTPUT, EVP_LIF_output, output_params, output_init_vars)
    output.set_extra_global_param("t_k",-1e5*np.ones(NUM_OUTPUT*p["N_MAX_SPIKE"],dtype=np.float32))
    output.set_extra_global_param("ImV",np.zeros(NUM_OUTPUT*p["N_MAX_SPIKE"],dtype=np.float32))

    if p["TRAIN"]:
        input.set_extra_global_param("spikeTimes", X_train)
        output.set_extra_global_param("label", Y_train)
    else:
        input.set_extra_global_param("spikeTimes", X_test)
        output.set_extra_global_param("label", Y_test)

    input_var_refs= {"startSpike": genn_model.create_var_ref(input, "startSpike"),
                     "last_startSpike": genn_model.create_var_ref(input, "last_startSpike"),
                     "back_startSpike": genn_model.create_var_ref(input, "back_startSpike"),
                     "back_endSpike": genn_model.create_var_ref(input, "back_endSpike"),
                     "back_spike": genn_model.create_var_ref(input, "back_spike"),
                     "rev_t": genn_model.create_var_ref(input, "rev_t")
                     }

    input_reset= model.add_custom_update("input_reset","neuronReset", EVP_input_reset, {}, {}, input_var_refs)
    
    hidden_var_refs= {"rp_ImV": genn_model.create_var_ref(hidden, "rp_ImV"),
                      "wp_ImV": genn_model.create_var_ref(hidden, "wp_ImV"),
                      "V": genn_model.create_var_ref(hidden, "V"),
                      "lambda_V": genn_model.create_var_ref(hidden, "lambda_V"),
                      "lambda_I": genn_model.create_var_ref(hidden, "lambda_I"),
                      "rev_t": genn_model.create_var_ref(hidden, "rev_t"),
                      "back_spike": genn_model.create_var_ref(hidden, "back_spike")
                      }
    hidden_reset=  model.add_custom_update("hidden_reset","neuronReset", EVP_neuron_reset, {"V_reset": p["V_RESET"], "N_max_spike": p["N_MAX_SPIKE"]}, {}, hidden_var_refs)

    output_reset_params= {"V_reset": p["V_RESET"],
                          "N_max_spike": p["N_MAX_SPIKE"],
                          "tau0": p["TAU_0"],
                          "tau1": p["TAU_1"]
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
                      "expsum": genn_model.create_var_ref(output, "expsum"),
                      "trial": genn_model.create_var_ref(output, "trial")                      
                      }
    output_reset=  model.add_custom_update("output_reset","neuronResetOutput", EVP_neuron_reset_output, output_reset_params, {}, output_var_refs)


    # synapse populations
    in_to_hid= model.add_synapse_population("in_to_hid", "DENSE_INDIVIDUALG", NO_DELAY, input, hidden, EVP_input_synapse,
                                            {}, in_to_hid_init_vars, {}, {}, "ExpCurr", {"tau": p["TAU_SYN"]}, {}
    )
    hid_to_out= model.add_synapse_population("hid_to_out", "DENSE_INDIVIDUALG", NO_DELAY, hidden, output, EVP_synapse,
                                             {}, hid_to_out_init_vars, {}, {}, "ExpCurr", {"tau": p["TAU_SYN"]}, {}
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
    hid_to_out.pre_target_var= "revIsyn"

    optimisers= [in_to_hid_learn, hid_to_out_learn]

    # enable buffered spike recording where desired
    for pop in p["REC_SPIKES"]:
        model.neuron_populations[pop].spike_recording_enabled= True
    
    if p["BUILD"]:
        model.build()
    model.load(num_recording_timesteps= p["SPK_REC_STEPS"])
    #print(dir(model))

    # ----------------------------------------------------------------------------
    # Simulation loop
    # ----------------------------------------------------------------------------

    if p["TRAIN"]:
        in_to_hid.pull_var_from_device("w")
        hid_to_out.pull_var_from_device("w")
        np.save(os.path.join(p["OUT_DIR"], "w_input_hidden_0.npy"), in_to_hid.vars["w"].view.copy())
        np.save(os.path.join(p["OUT_DIR"], "w_hidden_output_0.npy"), hid_to_out.vars["w"].view.copy())
    else:
        in_to_hid.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], "w_input_hidden_last.npy"))
        hid_to_out.vars["w"].view[:]= np.load(os.path.join(p["OUT_DIR"], "w_hidden_output_last.npy"))
        in_to_hid.push_var_to_device("w")
        hid_to_out.push_var_to_device("w")
    
    if p["TRAIN"]:
        N_trial= (p["N_TRAIN"] * N_CLASS) // p["N_BATCH"]
    else:
        N_trial= (p["N_TEST"] * N_CLASS) // p["N_BATCH"]

    good= 0.0    
    fst= output.vars["first_spike_t"].view
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
    Predict= []
    for epoch in range(p["N_EPOCH"]):
        #print(output.extra_global_params["label"].view[:])
        model.t= 0.0
        model.timestep= 0
        for var, val in input_init_vars.items():
            input.vars[var].view[:]= val
        input.push_state_to_device()
        for var, val in hidden_init_vars.items():
            hidden.vars[var].view[:]= val
        hidden.push_state_to_device()
        for var, val in output_init_vars.items():
            output.vars[var].view[:]= val
        output.push_state_to_device()
        for trial in range(N_trial):
            trial_end= (trial+1)*p["TRIAL_MS"]
            int_t= 0
            while (model.t < trial_end-1e-3*p["DT_MS"]):
                model.step_time()
                int_t += 1
                if len(p["REC_SPIKES"]) > 0:
                    if int_t%p["SPK_REC_STEPS"] == 0:
                        model.pull_recording_buffers_from_device()
                        for pop in p["REC_SPIKES"]:
                            the_pop= model.neuron_populations[pop]
                            if p["N_BATCH"] > 1:
                                spike_t[pop].append(the_pop.spike_recording_data[0][0]+epoch*N_trial*p["TRIAL_MS"])
                                spike_ID[pop].append(the_pop.spike_recording_data[0][1])
                            else:
                                spike_t[pop].append(the_pop.spike_recording_data[0]+epoch*N_trial*p["TRIAL_MS"])
                                spike_ID[pop].append(the_pop.spike_recording_data[1])

                for pop, var in p["REC_NEURONS"]:
                    the_pop= model.neuron_populations[pop]
                    the_pop.pull_var_from_device(var)
                    rec_vars_n[var+pop].append(the_pop.vars[var].view.copy())

                for pop, var in p["REC_SYNAPSES"]:
                    the_pop= model.synapse_populations[pop]
                    the_pop.pull_var_from_device(var)
                    rec_vars_s[var+pop].append(the_pop.vars[var].view.copy())
                                           
            if p["TRAIN"]:
                update_adam(learning_rate, adam_step, optimisers)
                adam_step += 1
                model.custom_update("EVPReduce")
                model.custom_update("EVPLearn")
                # do some checks and measure training error
                """
                output.pull_var_from_device("first_spike_t")
                model.pull_recording_buffers_from_device()
                if (p["N_BATCH"] > 1):
                    tt= np.vstack([x[0] for x in input.spike_recording_data])
                    ii= np.vstack([x[1] for x in input.spike_recording_data])
                else:
                    tt= input.spike_recording_data[0]
                    ii= input.spike_recording_data[1]
                #print(tt)
                #print(ii)
                xspkt= tt[ii == 1]
                X.append(xspkt-trial*p["TRIAL_MS"])
                yspkt= tt[ii == 2]
                Y.append(yspkt-trial*p["TRIAL_MS"])
                LB.append(Y_train[trial*p["N_BATCH"]:(trial+1)*p["N_BATCH"]])
                """
            else:
                output.pull_var_from_device("first_spike_t");
                #print(fst.shape)
                pred= np.argmin(fst,axis=-1)
                #print(pred.shape)
                good += np.sum(cnt[pred == Y_test[trial*p["N_BATCH"]:(trial+1)*p["N_BATCH"]]])
                #print(good)
                Predict.append(pred)
            model.custom_update("neuronReset")
            in_to_hid.in_syn[:]= 0.0
            in_to_hid.push_in_syn_to_device()
            hid_to_out.in_syn[:]= 0.0
            hid_to_out.push_in_syn_to_device()
            model.custom_update("neuronResetOutput")

            if trial % p["W_REPORT_INTERVAL"] == 0:
                in_to_hid.pull_var_from_device("w")
                np.save(os.path.join(p["OUT_DIR"], "w_input_hidden_e{}_t{}.npy".format(epoch,trial)), in_to_hid.vars["w"].view.copy())
                hid_to_out.pull_var_from_device("w")
                np.save(os.path.join(p["OUT_DIR"], "w_hidden_output_e{}_t{}.npy".format(epoch,trial)), hid_to_out.vars["w"].view.copy())

    """
    print(X)
    print(Y)
    print(LB)
    plt.figure()
    plt.scatter(X,Y,c=LB,s=0.5)
    plt.show()
    """
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
            
    if not p["TRAIN"]:
        print("Correct: {}".format(good/(N_trial*p["N_BATCH"])))
        Predict= np.hstack(Predict)
        print(Predict.shape)
        plt.figure()
        plt.scatter(X_t_orig[:,0],X_t_orig[:,1],c=Y_test,s=0.5)
        plt.figure()
        plt.scatter(X_t_orig[:,0],X_t_orig[:,1],c=Predict,s=0.5)
        plt.show()
        
    in_to_hid.pull_var_from_device("w")
    hid_to_out.pull_var_from_device("w")
    np.save(os.path.join(p["OUT_DIR"], "w_input_hidden_last.npy"), in_to_hid.vars["w"].view.copy())
    np.save(os.path.join(p["OUT_DIR"], "w_hidden_output_last.npy"), hid_to_out.vars["w"].view.copy())
    return (spike_t, spike_ID, rec_vars_n, rec_vars_s)

if __name__ == "__main__":
    run_yingyang(p)
