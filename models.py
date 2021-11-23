"""
LIF neuron for "eventprop" exact gradient descent in spiking neural networks as 
described in 
Timo C. Wunderlich & Christian Pehle, Event‑based backpropagation
can compute exact gradients for spiking neural networks, Scientific Reports (2021) 11:12829, https://doi.org/10.1038/s41598-021-91786-z

We use "EVP" in naming to indicate "eventprop".
"""
"""
from pygenn import genn_model 


The neuron model contains the variables and code for both, the forward and 
backward pass which will be executed simultaneously.
The intended usage is that the model is run in the forward pass for time T, and the backward pass from the previous input at the same time. For the very first input, this backward pass will not  be meaningful but as there are no stored spikes, it should just run empty and lead to zero updates on weights.
We will store (I-V)(t_k) at spike times in an egp. Similarly the relevant spike times are stored in an egp. to avoid running back beyond the first spike stored, we will prefill one entry with large, negative spike time.

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
    param_names=["tau_m","V_tresh","V_reset"],
    var_name_types=[("V", "scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),
                    ("rp_ImV","int"),("wp_ImV","int"),("back_spike","bool")],
    extra_global_params=[("t_k","scalar *"),("ImV","scalar *")],
    additional_input_vars=[("revIsyn", "scalar", 0.0)],
    sim_code="""
    // backward pass
    const scalar back_t= 2.0*$(rev_t)-$(t);
    $(lambda_V) -= $(lambda_V)/$(tau_m)*DT;
    $(lambda_I) += ($(lambda_V) - $(lambda_I))/$(tau_syn)*DT;
    if ($(back_spike)) {
        $(lambda_V) += 1.0/$(ImV)[$(id)*((int) $(N_max_spike))+$(rp_Im)]*($(V_thresh)*$(lambda_V) + $(revIsyn));
        // decrease read pointer (on ring buffer)
        $(rp_ImV)--;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(back_spike)= false;
    }    
    if (abs(back_t - $(t_k)[$(id)*((int) $(N_max_spike))+$(rp_ImV)]) < 1e-3*DT) {
        $(back_spike)= true;
    }
    // forward pass
    $(V) += ($(Isyn)-$(V))/$(tau_m)*DT;
    """,
    threshold_condition_code="""
    $(V) >= $(V_thres)
    """,
    reset_code="""
    // this is after a forward spike
    $(t_k)[$(id)*((int) $(N_max_spike))+$(wp_ImV)]= $(t);
    $(imV)[$(id)*((int) $(N_max_spike))+$(wp_ImV)]= $(Isyn)-$(V);
    $(wp_ImV)++;
    if ($(wp_ImV) >= ((int) $(N_max_spike))) $(wp_ImV)= 0;
    $(V)= $(V_reset);
    """,
    is_auto_refractory_required=False
)

# LIF neuron model for internal neurons (no dl_p/dt_k cost function term)
EVP_LIF_output = genn_model.create_custom_neuron_class(
    "EVP_LIF_output",
    param_names=["tau_m","V_tresh","V_reset","trial_t","tau0","tau1","alpha"],
    var_name_types=[("V", "scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),
                    ("rp_ImV","int"),("wp_ImV","int"),("back_spike","bool"),("first_spike_t","scalar"),("new_first_spike_t","scalar")],
    extra_global_params=[("t_k","scalar *"),("ImV","scalar *")],
    additional_input_vars=[("revIsyn", "scalar", 0.0)],
    sim_code="""
    // backward pass
    const scalar back_t= 2.0*$(rev_t)-$(t);
    $(lambda_V) -= $(lambda_V)/$(tau_m)*DT;
    $(lambda_I) += ($(lambda_V) - $(lambda_I))/$(tau_syn)*DT;
    if ($(back_spike)) {
        $(lambda_V) += 1.0/$(ImV)[$(id)*((int) $(N_max_spike))+$(rp_Im)]*($(V_thresh)*$(lambda_V) + $(revIsyn));
        if (abs($(t) - $(first_spike_t)) < 1e-3*DT) { 
            scalar fst= $(first_spike_t)-$(rev_t)+$(trial_t);
            if ($(id) == $(class)) {
                $(lambda_V) += (1.0-exp(-fst)/$(expsum))/$(tau0)+$(alpha)*$(tau1)*exp(fst/$(tau1));
            }
            else {
                $(lambda_V) -= exp(-fst)/$(expsum)/$(tau0);
            }
        // decrease read pointer (on ring buffer)
        $(rp_ImV)--;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(back_spike)= false;
    }    
    if (abs(back_t - $(t_k)[$(id)*((int) $(N_max_spike))+$(rp_ImV)]) < 1e-3*DT) {
        $(back_spike)= true;
    }
    // forward pass
    $(V) += ($(Isyn)-$(V))/$(tau_m)*DT;
    """,
    threshold_condition_code="""
    $(V) >= $(V_thres)
    """,
    reset_code="""
    // this is after a forward spike
    $(t_k)[$(id)*((int) $(N_max_spike))+$(wp_ImV)]= $(t);
    $(imV)[$(id)*((int) $(N_max_spike))+$(wp_ImV)]= $(Isyn)-$(V);
    $(wp_ImV)++;
    if ($(wp_ImV) >= ((int) $(N_max_spike))) $(wp_ImV)= 0;
    it ($(t) > $(new_first_spike_t)) $(new_first_spike_t)= $(t);
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
        dw+= $(lambda_I_post)
    """,
)

# custom update class for applying the gradient to synaptic weights
genn_model.create_custom_custom_update_class(
    "EVP_grad_application",
    var_name_types=[("reduced_dw", "scalar", VarAccess_REDUCE_BATCH_SUM)],
    var_refs=[("w", "scalar"),("dw", "scalar")],
    extra_global_params=[("eta", "scalar")],
    update_code="""
        $(reduced_dw) = $(dw);
        $(dw) = 0.0;
        $(w)+= $(eta)*$(tau)*$(reduced_dw);
    """
)

"""
This custom update class is for doing the switchover from one input to the next.
Note that V_reset should be the correct initial value for V as used to initialise the neuron model.
"""
    
# custom update class for resetting neurons at trial end
genn_model.create_custom_custom_update_class(
    "EVP_neuron_reset",
    param_names=["V_reset"],
    var_refs=[("rp_ImV","int"),("wp_ImV","int"),("V","scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),("back_spike","bool")],
    update_code= """
        $(rp_ImV)= $(wp_ImV)-1;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(rev_t)= $(t);
        $(lambda_V)= 0.0;
        $(lambda_I)= 0.0;
        $(V)= $(V_reset);
        $(back_spike)= false;
    """
)

# custom update class for resetting neurons at trial end for output neurons
genn_model.create_custom_custom_update_class(
    "EVP_neuron_reset_output",
    param_names=["V_reset","tau0","tau1"],
    var_refs=[("rp_ImV","int"),("wp_ImV","int"),("V","scalar"),("lambda_V","scalar"),("lambda_I","scalar"),("rev_t","scalar"),("back_spike","bool")],
    update_code= """
        // Yuck - super-hacky to do a population reduction, including hard-coded population size 3
        $(expsum)= 0.0;
        scalar *the_spike_t= &($(new_first_spike_t))-$(id);
        for (int i= 0; i < 3; i++) {
            $(expsum)+= exp(-(the_spike_t[i]-$(rev_t))/$(tau0));
        }
        $(rp_ImV)= $(wp_ImV)-1;
        if ($(rp_ImV) < 0) $(rp_ImV)= (int) $(N_max_spike)-1;
        $(rev_t)= $(t);
        $(lambda_V)= 0.0;
        $(lambda_I)= 0.0;
        $(V)= $(V_reset);
        $(back_spike)= false;
        $(first_spike_t)= $(new_first_spike_t);
        $(new_first_spike_t)= 1e-5;
    """
)
