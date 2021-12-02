"""
LIF neuron for "eventprop" exact gradient descent in spiking neural networks as 
described in 
Timo C. Wunderlich & Christian Pehle, Event‑based backpropagation
can compute exact gradients for spiking neural networks, Scientific Reports (2021) 11:12829, https://doi.org/10.1038/s41598-021-91786-z

We use "EVP" in naming to indicate "eventprop".
"""
from pygenn import genn_model 
from pygenn.genn_wrapper.Models import VarAccessDuplication_SHARED, VarAccess_REDUCE_BATCH_SUM, VarAccessMode_READ_ONLY, VarAccess_READ_ONLY_DUPLICATE

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
    """
)

"""
This custom update class is for doing the switchover from one input to the next for 
the input spike sources.
"""
    
# custom update class for resetting neurons at trial end
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

# custom update class for resetting neurons at trial end for output neurons
EVP_neuron_reset_output= genn_model.create_custom_custom_update_class(
    "EVP_neuron_reset_output",
    param_names=["V_reset","N_max_spike","tau0","tau1"],
    var_refs=[("rp_ImV","int"),("wp_ImV","int"),("V","scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),("back_spike","uint8_t"),("first_spike_t","scalar"),("new_first_spike_t","scalar"),("expsum","scalar"),("trial","unsigned int")],
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


#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------

# SpikeSourceArray as in standard GeNN but with a recording of spikes and backward
# spike triggering
EVP_SSA = genn_model.create_custom_neuron_class(
    "spikeSourceArray",
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
N_max_spike - maximum number of spikes

Extra Global Parameters:
t_k - spike times t_k of the neuron
ImV - (I-V)(t_k) where t_k is the kth spike

Extra input variables:
revIsyn - gets the reverse input from postsynaptic neurons
"""

# LIF neuron model for internal neurons (no dl_p/dt_k cost function term)
EVP_LIF = genn_model.create_custom_neuron_class(
    "EVP_LIF",
    param_names=["tau_m","V_thresh","V_reset","N_max_spike","tau_syn"],
    var_name_types=[("V", "scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),
                    ("rp_ImV","int"),("wp_ImV","int"),("back_spike","uint8_t")],
    extra_global_params=[("t_k","scalar*"),("ImV","scalar*")],
    additional_input_vars=[("revIsyn", "scalar", 0.0)],
    sim_code="""
    // backward pass
    const scalar back_t= 2.0*$(rev_t)-$(t)-DT;
    $(lambda_V) -= $(lambda_V)/$(tau_m)*DT;
    $(lambda_I) += ($(lambda_V) - $(lambda_I))/$(tau_syn)*DT;
    if ($(back_spike)) {
        //printf(\"%f\\n",$(revIsyn));
        $(lambda_V) += 1.0/$(ImV)[$(id)*((int) $(N_max_spike))+$(rp_ImV)]*($(V_thresh)*$(lambda_V) + $(revIsyn));
        // decrease read pointer (on ring buffer)
        $(rp_ImV)--;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(back_spike)= 0;
    }   
    // YUCK - need to trigger the back_spike the time step before to get the correct backward synaptic input
    if (abs(back_t - $(t_k)[$(id)*((int) $(N_max_spike))+$(rp_ImV)] - DT) < 1e-3*DT) {
        $(back_spike)= 1;
    }
    // forward pass
    $(V) += ($(Isyn)-$(V))/$(tau_m)*DT;
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    // this is after a forward spike
    $(t_k)[$(id)*((int) $(N_max_spike))+$(wp_ImV)]= $(t);
    $(ImV)[$(id)*((int) $(N_max_spike))+$(wp_ImV)]= $(Isyn)-$(V);
    $(wp_ImV)++;
    if ($(wp_ImV) >= ((int) $(N_max_spike))) $(wp_ImV)= 0;
    $(V)= $(V_reset);
    """,
    is_auto_refractory_required=False
)

# LIF neuron model for output neurons (includes contribution from dl_p/dt_k loss function term at jumps)
EVP_LIF_output = genn_model.create_custom_neuron_class(
    "EVP_LIF_output",
    param_names=["tau_m","V_thresh","V_reset","N_max_spike","tau_syn","trial_t","tau0","tau1","alpha","N_batch"],
    var_name_types=[("V", "scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),
                    ("rp_ImV","int"),("wp_ImV","int"),("back_spike","uint8_t"),("first_spike_t","scalar"),("new_first_spike_t","scalar"),("expsum","scalar"),("trial","unsigned int")],
    extra_global_params=[("t_k","scalar*"),("ImV","scalar*"),("label","int*")], 
    additional_input_vars=[("revIsyn", "scalar", 0.0)],
    sim_code="""
    // backward pass
    const scalar back_t= 2.0*$(rev_t)-$(t)-DT;
    $(lambda_V) -= $(lambda_V)/$(tau_m)*DT;
    $(lambda_I) += ($(lambda_V) - $(lambda_I))/$(tau_syn)*DT;
    //if ($(id) == 0) printf(\"%f:%f,%f,%f\\n\",$(t),$(first_spike_t),$(t_k)[$(id)*((int) $(N_max_spike))+$(rp_ImV)],back_t);
    if ($(back_spike)) {
        $(lambda_V) += 1.0/$(ImV)[$(id)*((int) $(N_max_spike))+$(rp_ImV)]*($(V_thresh)*$(lambda_V) + $(revIsyn));
        //if ($(id) == 0) printf(\"%f, %f\\n\",back_t,$(first_spike_t));
        if (abs(back_t - $(first_spike_t)) < 1e-2*DT) {
            scalar fst= $(first_spike_t)-$(rev_t)+$(trial_t);
            if ($(id) == $(label)[($(trial)-1)*(int)$(N_batch)+$(batch_id)]) {
            //if ($(id) == $(label)[($(trial)-1)*(int)$(N_batch)]) {
                scalar old_lambda= $(lambda_V);
                $(lambda_V) += ((1.0-exp(-fst/$(tau0))/$(expsum))/$(tau0)+$(alpha)/$(tau1)*exp(fst/$(tau1)))/$(N_batch);
                //printf(\"%g POS: Trial: %d, label: %d, ID: %d, expsum: %g, old: %g, new: %g\\n\",$(t),$(trial),$(label)[($(trial)-1)*(int)$(N_batch)],$(id),$(expsum),old_lambda,$(lambda_V));
            }
            else {
                scalar old_lambda= $(lambda_V);
                $(lambda_V) -= (exp(-fst/$(tau0))/$(expsum)/$(tau0))/$(N_batch);
                //printf(\"%g NEG: Trial: %d, label: %d, ID: %d, expsum: %g, old: %g, new: %g\\n\",$(t),$(trial),$(label)[($(trial)-1)*(int)$(N_batch)],$(id),$(expsum),old_lambda,$(lambda_V));
            }
        }
        // decrease read pointer (on ring buffer)
        $(rp_ImV)--;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(back_spike)= 0;
    }    
    // YUCK - need to trigger the back_spike the time step before to get the correct backward synaptic input
    if (abs(back_t - $(t_k)[$(id)*((int) $(N_max_spike))+$(rp_ImV)]-DT) < 1e-3*DT) {
        $(back_spike)= 1;
    }
    // forward pass
    $(V) += ($(Isyn)-$(V))/$(tau_m)*DT;
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    // this is after a forward spike
    $(t_k)[$(id)*((int) $(N_max_spike))+$(wp_ImV)]= $(t);
    $(ImV)[$(id)*((int) $(N_max_spike))+$(wp_ImV)]= $(Isyn)-$(V);
    $(wp_ImV)++;
    if ($(wp_ImV) >= ((int) $(N_max_spike))) $(wp_ImV)= 0;
    if ($(new_first_spike_t) < 0.0) $(new_first_spike_t)= $(t);
    $(V)= $(V_reset);
    """,
    is_auto_refractory_required=False
)

# synapses
EVP_synapse= genn_model.create_custom_weight_update_class(
    "EVP_synapse",
    var_name_types=[("w","scalar", VarAccessDuplication_SHARED),("dw","scalar")],
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
    "EVP_synapse",
    var_name_types=[("w","scalar", VarAccessDuplication_SHARED),("dw","scalar")],
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

