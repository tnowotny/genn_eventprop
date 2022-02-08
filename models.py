"""
LIF neuron for "eventprop" exact gradient descent in spiking neural networks as 
described in 
Timo C. Wunderlich & Christian Pehle, Event‑based backpropagation
can compute exact gradients for spiking neural networks, Scientific Reports (2021) 11:12829, https://doi.org/10.1038/s41598-021-91786-z

We use "EVP" in naming to indicate "eventprop".
"""
import numpy as np
from pygenn import genn_model 
from pygenn.genn_wrapper.Models import VarAccessDuplication_SHARED, VarAccess_REDUCE_BATCH_SUM, VarAccessMode_READ_ONLY, VarAccess_READ_ONLY, VarAccess_READ_ONLY_DUPLICATE

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------


# custom update class for reducing the gradient 
EVP_grad_reduce= genn_model.create_custom_custom_update_class(
    "EVP_grad_reduce",
    var_name_types=[("reduced_dw", "scalar", VarAccess_REDUCE_BATCH_SUM)],
    var_refs=[("dw", "scalar")],
    update_code="""
        $(reduced_dw) = $(dw);
        $(dw) = 0.0;
    """
)

# custom update to apply gradients using the Adam optimizer
adam_optimizer_model = genn_model.create_custom_custom_update_class(
    "adam_optimizer",
    param_names=["beta1", "beta2", "epsilon", "tau_syn", "N_batch"],
    var_name_types=[("m", "scalar"), ("v", "scalar")],
    extra_global_params=[("alpha", "scalar"), ("firstMomentScale", "scalar"),
                         ("secondMomentScale", "scalar")],
    var_refs=[("gradient", "scalar", VarAccessMode_READ_ONLY), ("variable", "scalar")],
    update_code="""
    scalar grad= -$(tau_syn)*$(gradient);
    // Update biased first moment estimate
    $(m) = ($(beta1) * $(m)) + ((1.0 - $(beta1)) * grad);
    // Update biased second moment estimate
    $(v) = ($(beta2) * $(v)) + ((1.0 - $(beta2)) * grad * grad);
    // Add gradient to variable, scaled by learning rate
    $(variable) -= ($(alpha) * $(m) * $(firstMomentScale)) / (sqrt($(v) * $(secondMomentScale)) + $(epsilon));
    //$(variable) -= $(alpha)*grad;
    """
)

normalize_model =genn_model.create_custom_custom_update_class(
    "normalize_by_expsum",
    param_names= [],
    extra_global_params=[("expsum", "scalar")],
    var_refs=[("w", "scalar")],
    update_code="""
    $(w) /= $(expsum);
    """
)

"""
This custom update class is for doing the switchover from one input to the next for 
the input spike sources.
"""
    
# custom update class for resetting input neurons at trial end YingYang
EVP_input_reset= genn_model.create_custom_custom_update_class(
    "EVP_input_reset",
    param_names=[],
    var_refs=[("startSpike","int"),("last_startSpike","int"),("back_startSpike","int"),("back_endSpike","int"),("back_spike","uint8_t"),("rev_t","scalar")],
    update_code= """
        $(back_endSpike)= $(last_startSpike)-1;
        $(last_startSpike)= $(startSpike);
        $(back_startSpike)= $(startSpike)-1;
        $(back_spike)= 0;
        $(rev_t)= $(t);
    """
)

# custom update class for resetting input neurons at trial end MNIST
EVP_input_reset_MNIST= genn_model.create_custom_custom_update_class(
    "EVP_input_reset_MNIST",
    param_names=["N_max_spike"],
    var_refs=[("rp_ImV","int"),("wp_ImV","int"),("back_spike","uint8_t"),("rev_t","scalar")],
    update_code= """
        $(rp_ImV)= $(wp_ImV)-1;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(back_spike)= 0;
        $(rev_t)= $(t);
    """
)

# custom update class for setting input neurons for fast shuffling to a new input
EVP_input_set_MNIST_shuffle= genn_model.create_custom_custom_update_class(
    "EVP_input_set_MNIST_shuffle",
    param_names=["N_batch", "num_input"],
    extra_global_params= [("allStartSpike", "int*"), ("allEndSpike", "int*"),("allInputID", "int*"),("trial","int")],
    var_refs=[("startSpike", "int"),("endSpike", "int")],
    update_code= """
        int myinid= $(allInputID)[$(trial)*((int) $(N_batch))+$(batch)];
        int myiid= myinid*((int) $(num_input))+$(id); 
        $(startSpike)= $(allStartSpike)[myiid];
        $(endSpike)= $(allEndSpike)[myiid];
        //printf("trial: %d, batch: %d, id: %d, myinid: %d, myiid: %d, start: %d, end: %d \\n", $(trial), $(batch), $(id), myinid, myiid, $(startSpike), $(endSpike));
    """
)
    
"""
This custom update class is for doing the switchover from one input to the next.
Note that V_reset should be the correct initial value for V as used to initialise the neuron model.
"""
    
# custom update class for resetting neurons at trial end
EVP_neuron_reset= genn_model.create_custom_custom_update_class(
    "EVP_neuron_reset",
    param_names=["V_reset","N_max_spike"],
    var_refs=[("rp_ImV","int"),("wp_ImV","int"),("V","scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),("back_spike","uint8_t")],
    update_code= """
        $(rp_ImV)= $(wp_ImV)-1;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(rev_t)= $(t);
        $(lambda_V)= 0.0;
        $(lambda_I)= 0.0;
        $(V)= $(V_reset);
        $(back_spike)= 0;
    """
)

# custom update class for resetting neurons at trial end with hidden layer rate normalisation terms
EVP_neuron_reset_reg= genn_model.create_custom_custom_update_class(
    "EVP_neuron_reset_reg",
    param_names=["V_reset","N_max_spike"],
    var_refs=[("rp_ImV","int"),("wp_ImV","int"),("V","scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),("back_spike","uint8_t"),("sNSum","scalar"),("new_sNSum","scalar"),("sNSum_all","scalar")],
    update_code= """
        // make a reduction of the spike number sum across the neuron population
        
        scalar sum= $(new_sNSum);
        sum += __shfl_xor_sync(0xFFFF, sum, 0x1);
        sum += __shfl_xor_sync(0xFFFF, sum, 0x2);
        sum += __shfl_xor_sync(0xFFFF, sum, 0x4);
        sum += __shfl_xor_sync(0xFFFF, sum, 0x8);
        sum += __shfl_xor_sync(0xFFFF, sum, 0x10);
        if (threadIdx.x%32 == 0) {
            atomicAdd(&(group->sNSum_all[0]),sum);
        }
        $(sNSum)= $(new_sNSum);
        $(new_sNSum)= 0.0;
        $(rp_ImV)= $(wp_ImV)-1;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(rev_t)= $(t);
        $(lambda_V)= 0.0;
        $(lambda_I)= 0.0;
        $(V)= $(V_reset);
        $(back_spike)= 0;
    """
)

# custom update class for resetting neurons at trial end for output neurons for YingYang
EVP_neuron_reset_output= genn_model.create_custom_custom_update_class(
    "EVP_neuron_reset_output",
    param_names=["V_reset","N_max_spike","tau0","tau1"],
    var_refs=[("rp_ImV","int"),("wp_ImV","int"),("V","scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),("back_spike","uint8_t"),("first_spike_t","scalar"),("new_first_spike_t","scalar"),("expsum","scalar"),("trial","int")],
    update_code= """
        scalar mexp;
        if ($(new_first_spike_t) > 0.0) {
            mexp= exp(-($(new_first_spike_t)-$(rev_t))/$(tau0));}
        else
            mexp= 0.0;
        //printf(\"%g, %d, %g, %g\\n\",$(t),$(id),$(new_first_spike_t),$(rev_t));
        #define __CUDA__
        #ifdef __CUDA__
        scalar sum= __shfl_sync(0x7, mexp, 0);
        sum+= __shfl_sync(0x7, mexp, 1);
        sum+= __shfl_sync(0x7, mexp, 2);
        $(expsum)= sum;
        #else
        // YUCK - terrible hack for CPU_ONLY
        if ($(id) == 0) {
             $(expsum)= 0.0;
             group->expsum[1]= 0.0;
             group->expsum[2]= 0.0;
        }
        group->expsum[0]+= mexp;
        if ($(id) == 2) {
           group->expsum[1]= group->expsum[0];
           $(expsum)= group->expsum[0];
        }
        #endif
        //printf(\"%g\\n\",$(expsum));
        //printf(\"ID: %d, rp: %d, wp: %d\\n\",$(id),$(rp_ImV),$(wp_ImV)); 
        $(rp_ImV)= $(wp_ImV)-1;
        if ($(rp_ImV) < 0) $(rp_ImV)= ((int) $(N_max_spike))-1;
        $(rev_t)= $(t);
        $(lambda_V)= 0.0;
        $(lambda_I)= 0.0;
        $(V)= $(V_reset);
        $(back_spike)= 0;
        $(first_spike_t)= $(new_first_spike_t);
        $(new_first_spike_t)= -1e5;
        $(trial)++;
    """
)

# custom update class for resetting output neurons at trial end for MNIST
EVP_neuron_reset_output_MNIST= genn_model.create_custom_custom_update_class(
    "EVP_neuron_reset_output_MNIST",
    param_names=["V_reset","N_class"],
    var_refs=[("max_V","scalar"),("new_max_V","scalar"),("max_t","scalar"),("new_max_t","scalar"),("V","scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),("expsum","scalar"),("exp_V","scalar"),("trial","int")],
    update_code= """
        scalar mexp= 0.0;
        #define __CUDA__
        #ifdef __CUDA__
        scalar m= 0.0;
        if ($(id) < $(N_class)) m= $(new_max_V);
        m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x1));
        m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x2));
        m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x4));
        m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x8));
        if ($(id) < $(N_class)) {
             mexp= exp($(new_max_V) - m);
             $(exp_V)= mexp;
        } else $(exp_V)= 0.0;
        mexp += __shfl_xor_sync(0xFFFF, mexp, 0x1);
        mexp += __shfl_xor_sync(0xFFFF, mexp, 0x2);
        mexp += __shfl_xor_sync(0xFFFF, mexp, 0x4);
        mexp += __shfl_xor_sync(0xFFFF, mexp, 0x8);
        $(expsum)= mexp;
        #else
        // YUCK - terrible hack for CPU_ONLY
        if ($(id) == 0) {
             $(expsum)= 0.0;
        }
        group->expsum[0]+= mexp;
        if ($(id) == 9) {
             for (int _i_= 1; _i_ < 10; _i_++) {
                 group->expsum[_i_]= group->expsum[0];
             }
        }
        #endif
        //printf(\"%g\\n\",$(expsum));
        $(rev_t)= $(t);
        $(lambda_V)= 0.0;
        $(lambda_I)= 0.0;
        $(V)= $(V_reset);
        $(max_V)= $(new_max_V);
        $(max_t)= $(new_max_t);
        $(new_max_V)= $(V_reset);
        $(new_max_t)= $(t);
        $(trial)++;
    """
)

# custom update class for resetting output neurons at trial end for SHD
# almost like MNIST but annoyingly more classes/ output neurons
EVP_neuron_reset_output_SHD= genn_model.create_custom_custom_update_class(
    "EVP_neuron_reset_output_SHD",
    param_names=["V_reset","N_class"],
    var_refs=[("max_V","scalar"),("new_max_V","scalar"),("max_t","scalar"),("new_max_t","scalar"),("V","scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),("expsum","scalar"),("exp_V","scalar"),("trial","int")],
    update_code= """
        scalar mexp= 0.0;
        #define __CUDA__
        #ifdef __CUDA__
        scalar m= 0.0;
        if ($(id) < $(N_class)) m= $(new_max_V);
        m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x1));
        m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x2));
        m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x4));
        m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x8));
        m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x10));
        if ($(id) < $(N_class)) {
             mexp= exp($(new_max_V) - m);
             $(exp_V)= mexp;
        } else $(exp_V)= 0.0;
        mexp += __shfl_xor_sync(0xFFFF, mexp, 0x1);
        mexp += __shfl_xor_sync(0xFFFF, mexp, 0x2);
        mexp += __shfl_xor_sync(0xFFFF, mexp, 0x4);
        mexp += __shfl_xor_sync(0xFFFF, mexp, 0x8);
        mexp += __shfl_xor_sync(0xFFFF, mexp, 0x10);
        $(expsum)= mexp;
        #else
        // YUCK - terrible hack for CPU_ONLY
        if ($(id) == 0) {
             $(expsum)= 0.0;
        }
        group->expsum[0]+= mexp;
        if ($(id) == 19) {
             for (int _i_= 1; _i_ < 20; _i_++) {
                 group->expsum[_i_]= group->expsum[0];
             }
        }
        #endif
        //printf(\"%g\\n\",$(expsum));
        $(rev_t)= $(t);
        $(lambda_V)= 0.0;
        $(lambda_I)= 0.0;
        $(V)= $(V_reset);
        $(max_V)= $(new_max_V);
        $(max_t)= $(new_max_t);
        $(new_max_V)= $(V_reset);
        $(new_max_t)= $(t);
        $(trial)++;
    """
)

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------

"""
 SpikeSourceArray as in standard GeNN but with a recording of spikes and backward
 spike triggering
"""
EVP_SSA = genn_model.create_custom_neuron_class(
    "EVP_spikeSourceArray",
    param_names=[],
    var_name_types=[("startSpike", "int"), ("endSpike", "int", VarAccess_READ_ONLY_DUPLICATE),("back_spike","uint8_t"), ("last_startSpike","int"), ("back_endSpike","int"),
                    ("back_startSpike","int"),("rev_t","scalar")],
    sim_code= """
        // backward pass
        const scalar back_t= 2.0*$(rev_t)-$(t)-DT;
        // YUCK - need to trigger the back_spike the time step before 
        if (($(back_startSpike) != $(back_endSpike)) && (back_t-DT < $(spikeTimes)[$(back_startSpike)] + DT)) {
            $(back_spike)= 1;
            $(back_startSpike)--;
        }
        else $(back_spike)= 0;
    """,
    threshold_condition_code= """
        $(startSpike) != $(endSpike) && 
        $(t) >= $(spikeTimes)[$(startSpike)] 
    """,
    reset_code= """
        $(startSpike)++;\n
    """,
    extra_global_params= [("spikeTimes", "scalar*")],
    is_auto_refractory_required=False
)


"""
A type of SpikeSourceArray that provides more efficient updating of input patterns when shuffling of inputs
is required. the basic idea here is that spike patterns are pre-loaded into the spikeTimes array but each pattern
assumes a start at time 0 and patterns are not "concatenated" for each neuron. Then, there is a matching startSpike
and endSpike array pointing into the right positions for each neuron for each input pattern. By indexing into the startspike and endSpike array, one can choose an input pattern (0 to #patterns-1).
As above, for the MNIST experiment, we need "dropout" or "unreliable spiking", which can be switched on (training) and off (testing)
"""
EVP_SSA_MNIST_SHUFFLE = genn_model.create_custom_neuron_class(
    "EVP_spikeSourceArray_MNIST_Shuffle",
    param_names=["N_neurons","N_max_spike"],
    var_name_types=[("startSpike", "int"), ("endSpike", "int", VarAccess_READ_ONLY_DUPLICATE), ("back_spike","uint8_t"), ("rp_ImV","int"),("wp_ImV","int"),("rev_t","scalar")],
    sim_code= """
        int buf_idx= $(batch)*((int) $(N_neurons))*((int) $(N_max_spike))+$(id)*((int) $(N_max_spike));
        // backward pass
        const scalar back_t= 2.0*$(rev_t)-$(t)-DT;
        if ($(back_spike)) {
            // decrease read pointer (on ring buffer)
            $(rp_ImV)--;
            if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
            $(back_spike) = 0;
        }
        // YUCK - need to trigger the back_spike the time step before 
        if (abs(back_t - $(t_k)[buf_idx+$(rp_ImV)] - DT) < 1e-3*DT) {
            $(back_spike)= 1;
        }
        // forward spikes
        //printf("%d, %d \\n", $(startSpike), $(endSpike));
        //if ($(startSpike) != $(endSpike)) {
        //    printf("t= %f, sT= %f \\n",$(t), $(spikeTimes)[$(startSpike)]);
        //}
        if ($(startSpike) != $(endSpike) && ($(t) >= $(t_offset)+$(spikeTimes)[$(startSpike)]+DT))
             $(startSpike)++;
    """,
    threshold_condition_code= """
        $(startSpike) != $(endSpike) && 
        $(t) >= $(t_offset)+$(spikeTimes)[$(startSpike)] &&
        $(gennrand_uniform) > $(pDrop)
    """,
    reset_code= """
        // this is after a forward spike
        $(t_k)[buf_idx+$(wp_ImV)]= $(t);
        $(wp_ImV)++;
        if ($(wp_ImV) >= ((int) $(N_max_spike))) $(wp_ImV)= 0;
    """,
    extra_global_params= [("spikeTimes", "scalar*"), ("t_offset","scalar"), ("t_k", "scalar*"),("pDrop", "scalar")],
    is_auto_refractory_required=False
)


"""
The neuron model contains the variables and code for both, the forward and 
backward pass which will be executed simultaneously.
The intended usage is that the model is run in the forward pass for time T, and the backward pass from the previous input at the same time. For the very first input, this backward pass will not  be meaningful but as there are no stored spikes, it should just run empty and lead to zero updates on weights.
We will store (I-V)(t_k) at spike times in an egp. Similarly the relevant spike times are stored in an egp. to avoid running back beyond the first spike stored, we will prefill one entry with large, negative spike time.
Note that this is intended to be operated in continuous time (no reset of t for each input) and there are some subtle dependencies on this, most prominently, the ring buffers for ImV and t_k do not have any explicit boundary checking. On the baclward pass we depend on the fact that earlier, irrelevant spikes that are still in the buffer are at time values that are not reached by the backward time stepping. If resetting time, this would not work.

Variables:
forward:
V - membrane potential
backward:
lambda_V - Lagrange multiplier for V
lambda_I - Lagrange multiplier for I($(t)-$(rev_t))
control:
rp_ImV - read pointer into ImV and t_k
wp_ImV - write pointer into ImV and t_k

Parameters:
tau_m - membrane potential time constant
tau_syn - synaptic times constant
V_thresh - spiking threshold
trial_t - times for a trial (input presentation) = duration of a pass
N_neurons - number of neurons in population
N_max_spike - maximum number of spikes

Extra Global Parameters:
t_k - spike times t_k of the neuron
ImV - (I-V)(t_k) where t_k is the kth spike

Extra input variables:
revIsyn - gets the reverse input from postsynaptic neurons
"""

# LIF neuron model for internal neurons for both YingYang and MNIST tasks (no dl_p/dt_k cost function term or jumps on max voltage)
EVP_LIF = genn_model.create_custom_neuron_class(
    "EVP_LIF",
    param_names=["tau_m","V_thresh","V_reset","N_neurons","N_max_spike","tau_syn"],
    var_name_types=[("V", "scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),
                    ("rp_ImV","int"),("wp_ImV","int"),("back_spike","uint8_t"),("lambda_jump","scalar")],
    extra_global_params=[("t_k","scalar*"),("ImV","scalar*")],
    additional_input_vars=[("revIsyn", "scalar", 0.0)],
    sim_code="""
    int buf_idx= $(batch)*((int) $(N_neurons))*((int) $(N_max_spike))+$(id)*((int) $(N_max_spike));
    // backward pass
    const scalar back_t= 2.0*$(rev_t)-$(t)-DT;
    //$(lambda_V) -= $(lambda_V)/$(tau_m)*DT;
    //$(lambda_I) += ($(lambda_V) - $(lambda_I))/$(tau_syn)*DT;
    $(lambda_I)= $(tau_m)/($(tau_syn)-$(tau_m))*$(lambda_V)*(exp(-DT/$(tau_syn))-exp(-DT/$(tau_m)))+$(lambda_I)*exp(-DT/$(tau_syn));
    $(lambda_V)= $(lambda_V)*exp(-DT/$(tau_m));
    if ($(back_spike)) {
        //printf(\"%f\\n",$(revIsyn));
    
        $(lambda_jump)= 1.0/$(ImV)[buf_idx+$(rp_ImV)]*($(V_thresh)*$(lambda_V) + $(revIsyn)); // for debugging only
        $(lambda_V) += 1.0/$(ImV)[buf_idx+$(rp_ImV)]*($(V_thresh)*$(lambda_V) + $(revIsyn));
        // decrease read pointer (on ring buffer)
        $(rp_ImV)--;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(back_spike)= 0;
    }   
    // YUCK - need to trigger the back_spike the time step before to get the correct backward synaptic input
    if (abs(back_t - $(t_k)[buf_idx+$(rp_ImV)] - DT) < 1e-3*DT) {
        $(back_spike)= 1;
    }
    // forward pass
    //$(V) += ($(Isyn)-$(V))/$(tau_m)*DT;  // simple Euler
    $(V)= $(tau_syn)/($(tau_m)-$(tau_syn))*$(Isyn)*(exp(-DT/$(tau_m))-exp(-DT/$(tau_syn)))+$(V)*exp(-DT/$(tau_m));   // exact solution
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    // this is after a forward spike
    $(t_k)[buf_idx+$(wp_ImV)]= $(t);
    $(ImV)[buf_idx+$(wp_ImV)]= $(Isyn)-$(V);
    $(wp_ImV)++;
    if ($(wp_ImV) >= ((int) $(N_max_spike))) $(wp_ImV)= 0;
    $(V)= $(V_reset);
    """,
    is_auto_refractory_required=False
)

# LIF neuron model for internal neurons for SHD task with regularisation - which introduced dlp/dtk type terms
EVP_LIF_reg = genn_model.create_custom_neuron_class(
    "EVP_LIF",
    param_names=["tau_m","V_thresh","V_reset","N_neurons","N_max_spike","tau_syn"],
    var_name_types=[("V", "scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),
                    ("rp_ImV","int"),("wp_ImV","int"),("back_spike","uint8_t"),("lambda_jump","scalar"),("sNSum","scalar"),("new_sNSum","scalar"),("sNSum_all","scalar")],
    # TODO: should the sNSum variable be integers? Would it conflict with the atomicAdd? also , will this work for double precision (atomicAdd?)?
    extra_global_params=[("t_k","scalar*"),("ImV","scalar*"),("nu_lower","scalar"),("nu_upper","scalar")],
    additional_input_vars=[("revIsyn", "scalar", 0.0)],
    sim_code="""
    int buf_idx= $(batch)*((int) $(N_neurons))*((int) $(N_max_spike))+$(id)*((int) $(N_max_spike));
    // backward pass
    const scalar back_t= 2.0*$(rev_t)-$(t)-DT;
    //$(lambda_V) -= $(lambda_V)/$(tau_m)*DT;
    //$(lambda_I) += ($(lambda_V) - $(lambda_I))/$(tau_syn)*DT;
    $(lambda_I)= $(tau_m)/($(tau_syn)-$(tau_m))*$(lambda_V)*(exp(-DT/$(tau_syn))-exp(-DT/$(tau_m)))+$(lambda_I)*exp(-DT/$(tau_syn));
    $(lambda_V)= $(lambda_V)*exp(-DT/$(tau_m));
    if ($(back_spike)) {
        //printf(\"%f\\n",$(revIsyn));
    
        $(lambda_jump)= 1.0/$(ImV)[buf_idx+$(rp_ImV)]*($(V_thresh)*$(lambda_V) + $(revIsyn)); // for debugging only
        $(lambda_V) += 1.0/$(ImV)[buf_idx+$(rp_ImV)]*($(V_thresh)*$(lambda_V) + $(revIsyn));
        // decrease read pointer (on ring buffer)
        $(rp_ImV)--;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        // contributions from regularisation
        $(back_spike)= 0;
    }   
    // YUCK - need to trigger the back_spike the time step before to get the correct backward synaptic input
    if (abs(back_t - $(t_k)[buf_idx+$(rp_ImV)] - DT) < 1e-3*DT) {
        $(back_spike)= 1;
    }
    // forward pass
    //$(V) += ($(Isyn)-$(V))/$(tau_m)*DT;  // simple Euler
    $(V)= $(tau_syn)/($(tau_m)-$(tau_syn))*$(Isyn)*(exp(-DT/$(tau_m))-exp(-DT/$(tau_syn)))+$(V)*exp(-DT/$(tau_m));   // exact solution
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    // this is after a forward spike
    $(t_k)[buf_idx+$(wp_ImV)]= $(t);
    $(ImV)[buf_idx+$(wp_ImV)]= $(Isyn)-$(V);
    $(wp_ImV)++;
    if ($(wp_ImV) >= ((int) $(N_max_spike))) $(wp_ImV)= 0;
    $(V)= $(V_reset);
    """,
    is_auto_refractory_required=False
)

# LIF neuron model for output neurons of YingYang task (includes contribution from dl_p/dt_k loss function term at jumps)
EVP_LIF_output = genn_model.create_custom_neuron_class(
    "EVP_LIF_output",
    param_names=["tau_m","V_thresh","V_reset","N_neurons","N_max_spike","tau_syn","trial_t","tau0","tau1","alpha","N_batch"],
    var_name_types=[("V", "scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),
                    ("rp_ImV","int"),("wp_ImV","int"),("back_spike","uint8_t"),
                    ("first_spike_t","scalar"),("new_first_spike_t","scalar"),("expsum","scalar"),
                    ("trial","int"),("lambda_jump","scalar")],
    extra_global_params=[("t_k","scalar*"),("ImV","scalar*"),("label","int*")], 
    additional_input_vars=[("revIsyn", "scalar", 0.0)],
    sim_code="""
    int buf_idx= $(batch)*((int) $(N_neurons))*((int) $(N_max_spike))+$(id)*((int) $(N_max_spike));    
    // backward pass
    const scalar back_t= 2.0*$(rev_t)-$(t)-DT;
    //$(lambda_V) -= $(lambda_V)/$(tau_m)*DT;
    //$(lambda_I) += ($(lambda_V) - $(lambda_I))/$(tau_syn)*DT;
    $(lambda_I)= $(tau_m)/($(tau_syn)-$(tau_m))*$(lambda_V)*(exp(-DT/$(tau_syn))-exp(-DT/$(tau_m)))+$(lambda_I)*exp(-DT/$(tau_syn));
    $(lambda_V)= $(lambda_V)*exp(-DT/$(tau_m));
    //if ($(id) == 0) printf(\"%f:%f,%f,%f\\n\",$(t),$(first_spike_t),$(t_k)[buf_idx+$(rp_ImV)],back_t);
    if ($(back_spike)) {
        if ($(first_spike_t) < 0.0) {// we are dealing with a "phantom spike" introduced because the correct neuron did not spike
            scalar fst= $(trial_t);
            //printf(\"adding %f\\n\",$(alpha)/((1.05*$(trial_t)-fst)*(1.05*$(trial_t)-fst))/$(N_batch));
            //$(lambda_V) += $(alpha)/$(tau1)*exp(fst/$(tau1))/$(N_batch);
            $(lambda_V) += $(alpha)/((1.01*$(trial_t)-fst)*(1.01*$(trial_t)-fst))/$(N_batch);
        }
        else {
            $(lambda_jump)= 1.0/$(ImV)[buf_idx+$(rp_ImV)]*($(V_thresh)*$(lambda_V) + $(revIsyn)); // for debugging only
            $(lambda_V) += 1.0/$(ImV)[buf_idx+$(rp_ImV)]*($(V_thresh)*$(lambda_V) + $(revIsyn));
            //if (back_t - $(first_spike_t) <= -1e-2*DT) printf("back_t: %e, fst: %e",back_t,$(first_spike_t)); 
            assert(back_t - $(first_spike_t) > -1e-2*DT);
            if (abs(back_t - $(first_spike_t)) < 1e-2*DT) {
                scalar fst= $(first_spike_t)-$(rev_t)+$(trial_t);
                if ($(id) == $(label)[($(trial)-1)*(int)$(N_batch)+$(batch)]) {
                    //$(lambda_V) += ((1.0-exp(-fst/$(tau0))/$(expsum))/$(tau0)+$(alpha)/$(tau1)*exp(fst/$(tau1)))/$(N_batch);
                    $(lambda_V) += 1.0/$(ImV)[buf_idx+$(rp_ImV)]*((1.0-exp(-fst/$(tau0))/$(expsum))/$(tau0)+$(alpha)/((1.01*$(trial_t)-fst)*(1.01*$(trial_t)-fst)))/$(N_batch);
                    //$(lambda_V) += ((1.0-exp(-fst/$(tau0))/$(expsum))/$(tau0))/$(N_batch);
                }
                else {
                    $(lambda_V) -= 1.0/$(ImV)[buf_idx+$(rp_ImV)]*(exp(-fst/$(tau0))/$(expsum)/$(tau0))/$(N_batch);
                }
            }
            // decrease read pointer (on ring buffer)
            $(rp_ImV)--;
            if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        }
        $(back_spike)= 0;
    }    
    // do this only from trial 1 onwards (i.e. do not try to do backward pass in trial 0)
    // YUCK - need to trigger the back_spike the time step before to get the correct backward synaptic input
    // YUCKYUCK - need to trigger a pretend back_spike if no spike occurred to keep in operating regime
    if (($(trial) > 0) && ((abs(back_t - $(t_k)[buf_idx+$(rp_ImV)]-DT) < 1e-3*DT) || (($(t) == $(rev_t)) && ($(first_spike_t) < 0.0) && $(id) == $(label)[($(trial)-1)*(int)$(N_batch)+$(batch)]))) {
        $(back_spike)= 1;
    }
    // forward pass
    //$(V) += ($(Isyn)-$(V))/$(tau_m)*DT;   // simple Euler
    $(V)= $(tau_syn)/($(tau_m)-$(tau_syn))*$(Isyn)*(exp(-DT/$(tau_m))-exp(-DT/$(tau_syn)))+$(V)*exp(-DT/$(tau_m));    // exact solution
    """,
    threshold_condition_code="""
    ($(V) >= $(V_thresh))
    """,
    reset_code="""
    // this is after a forward spike
    $(t_k)[buf_idx+$(wp_ImV)]= $(t);
    $(ImV)[buf_idx+$(wp_ImV)]= $(Isyn)-$(V);
    $(wp_ImV)++;
    if ($(wp_ImV) >= ((int) $(N_max_spike))) $(wp_ImV)= 0;
    if ($(new_first_spike_t) < 0.0) $(new_first_spike_t)= $(t);
    $(V)= $(V_reset);
    """,
    is_auto_refractory_required=False
)

# LIF neuron model for output neurons in the MNIST task - non-spiking and jumps in backward
# pass at times where the voltage reaches its maximum
EVP_LIF_output_MNIST = genn_model.create_custom_neuron_class(
    "EVP_LIF_output_MINST",
    param_names=["tau_m","tau_syn","trial_t","N_batch"],
    var_name_types=[("V", "scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),
                    ("max_V","scalar"),("new_max_V","scalar"),
                    ("max_t","scalar"),("new_max_t","scalar"),("expsum","scalar"),("exp_V","scalar"),
                    ("trial","int"),("lambda_jump","scalar")],
    extra_global_params=[("t_k","scalar*"),("ImV","scalar*"),("label","int*")], 
    sim_code="""
    // backward pass
    const scalar back_t= 2.0*$(rev_t)-$(t)-DT;
    //$(lambda_V) -= $(lambda_V)/$(tau_m)*DT;  // simple Euler
    //$(lambda_I) += ($(lambda_V) - $(lambda_I))/$(tau_syn)*DT;  // simple Euler
    $(lambda_I)= $(tau_m)/($(tau_syn)-$(tau_m))*$(lambda_V)*(exp(-DT/$(tau_syn))-exp(-DT/$(tau_m)))+$(lambda_I)*exp(-DT/$(tau_syn));
    $(lambda_V)= $(lambda_V)*exp(-DT/$(tau_m));
    if (($(trial) > 0) && (abs(back_t - $(max_t)) < 1e-3*DT)) {
        //if (($(id) == 0) && ($(batch) == 0) && ($(trial) < 11)) printf("%d\\n",$(label)[($(trial)-1)*(int)$(N_batch)+$(batch)]);
        if ($(id) == $(label)[($(trial)-1)*(int)$(N_batch)+$(batch)]) {
            $(lambda_V) += ((1.0-$(exp_V)/$(expsum))/$(tau_m))/$(N_batch);
            //printf("%d. up: label= %d, lambda_V(l(i))= %g \\n", ($(trial)-1)*(int)$(N_batch)+$(batch), $(label)[($(trial)-1)*(int)$(N_batch)+$(batch)], $(lambda_V));
        }
        else {
            $(lambda_V) -= ($(exp_V)/$(expsum)/$(tau_m))/$(N_batch);
            //printf("%d. down: label= %d, lambda_Vk= %g \\n", ($(trial)-1)*(int)$(N_batch)+$(batch), $(label)[($(trial)-1)*(int)$(N_batch)+$(batch)], $(lambda_V));        
        }
    }    
    // forward pass
    // update the maximum voltage
    if ($(V) > $(new_max_V)) {
        $(new_max_t)= $(t);
        $(new_max_V)= $(V);
    }
    //$(V) += ($(Isyn)-$(V))/$(tau_m)*DT;   // simple Euler
    $(V)= $(tau_syn)/($(tau_m)-$(tau_syn))*$(Isyn)*(exp(-DT/$(tau_m))-exp(-DT/$(tau_syn)))+$(V)*exp(-DT/$(tau_m));    // exact solution
    """,
    threshold_condition_code="",
    reset_code="",
    is_auto_refractory_required=False
)

# synapses
EVP_synapse= genn_model.create_custom_weight_update_class(
    "EVP_synapse",
    var_name_types=[("w","scalar", VarAccess_READ_ONLY),("dw","scalar")],
    sim_code="""
        $(addToInSyn, $(w));
    """,
    event_threshold_condition_code="""
       $(back_spike_pre)
    """,
    event_code="""
        $(addToPre, $(w)*($(lambda_V_post)-$(lambda_I_post)));
        $(dw)+= $(lambda_I_post);
    """,
)

EVP_input_synapse= genn_model.create_custom_weight_update_class(
    "EVP_input_synapse",
    var_name_types=[("w","scalar", VarAccess_READ_ONLY),("dw","scalar")],
    sim_code="""
        $(addToInSyn, $(w));
    """,
    event_threshold_condition_code="""
       $(back_spike_pre)
    """,
    event_code="""
        $(dw)+= $(lambda_I_post);
    """,
)

my_Exp_Curr= genn_model.create_custom_postsynaptic_class(
    "my_Exp_Curr",
    param_names=["tau"],
    decay_code="$(inSyn) *= $(expDecay);",
    apply_input_code="$(Isyn) += $(inSyn);",
    derived_params=[("expDecay", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())]
)
