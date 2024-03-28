# genn_eventprop
Implementation of eventprop learning in GeNN


The repository contains research code that was used to make a first implementation of the Eventprop learning rule [(Wunderlich and Pehle 2021)](https://doi.org/10.1038/s41598-021-91786-z) and test it against increasingly difficult benchmarks. The code underlies the publication: Nowotny, T., Turner, J.P. and Knight, J.C., 2022. Loss shaping enhances exact gradient learning with eventprop in spiking neural networks. arXiv preprint [arXiv:2212.01232](https://arxiv.org/abs/2212.01232).

The code is generally organised so that the GeNN model definitions are colected in `models.py` and for the benchmarks there are individual python files `simulator_XXX.py` as follows:

- `simulator_yinyang.py`: Train to classify the Yinyang benchmark dataset [(Kriener et al. 2022)](https://doi.org/10.1145/3517343.3517380)
- `simulator_MNIST.py`: Train to classify the MNIST dataset [(Lecun et al. 1998)](https://doi.org/10.1109/5.726791)
- `simulator_SHD.py`: Train to classify the Spikeing Heidelberg Digits (SHD) [(Cramer et al. 2020)](https://doi.org/10.1109/TNNLS.2020.3044364) or Spiking Google Speech Commands (SSC), a spiking version of Google Speech Commands [(Warden 2018)](https://doi.org/10.48550/arXiv.1804.03209) datasets. Which to train is selected with parameter `p["DATASET"]`
- `simulator_DVS_gesture.py`: *experimental*, train to classify the DVS gesture dataset [(Amir et al. 2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html)
- `simulator_SMNIST.py`: *experimental*, train to classify the sequential MNIST dataset [(Quoc et al. 2015)](https://doi.org/10.48550/arXiv.1504.00941), [(Costa et al. 2017)](https://proceedings.neurips.cc/paper/2017/hash/45fbc6d3e05ebd93369ce542e8f2322d-Abstract.html)

The latter two simulators are in an early experimental stage and did not contribute to the publication.

Please note that this is research code and comes with no warranty of fitness for any purpose. Also, the lessons learned with this code are incorporated into the [ml-genn](https://github.com/genn-team/ml_genn) that is better suitable for third party use.

## Details
Each of the `simulator_XXX.py` files contains the definition of a simulator class that can then be used to train, test or run cross-validation on the benchmark datasets in question. An example for a simple script to do so is `train_SHD.py`.

The behaviour of the simulator class is fine-tuned by a dictionary `p` of parameters that is passed to the methods. It is defined in the `simulator_XXX.py` files with standard values and can then be adjusted to the intended values, e.g.

    from simulator_SHD import *
    p["TRAIL_MS"] = 100
    mn = SHD_model(p)
    mn.train(p)

All potential parameters used in these dictionaries are detailed in the tables below.

## Yinyang parameters
### General parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| OUT_DIR   | Directory where to write results               | "." (current dir) |
| DT_MS     | Time step od the simulations in millisceonds   | 0.1 |
| BUILD     | Whether to (re)build the GeNN model, can be set to False if there are repeated runs of the same model | True |
| TIMING   | Whether to record timing information during a run | True |

### Experiment parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| TRAIN_DATA_SEED | Seed for the random number generator used for generating training data | 123 |
| TEST_DATA_SEED | Seed for the random number generator used for generating testing data | 456 |
| TRIAL_MS | Duration of a trial in milliseconds | 30.0 |
| N_MAX_SPIKE | Size of the bufferfor saved spikes in number of spikes; note the buffer contains saved spikes across two trials | 400 |
| N_BATCH | Size of mini-batches run in parallel | 32 |
| N_TRAIN | Number of training examples | 1000 mini-batches |
| N_EPOCHS | Number of epochs to train | 10 |
| N_TEST | Number of test examples | 25 mini-batches |

### Network structure
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| NUM_HIDDEN | Number of hidden neurons                      | 200     |

### Model parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| TAU_SYN   | Synaptic timescale in millisecons              | 5.0     |
| TAU_MEM   | Membrane times constant in milliseconds        | 20.0    |
| V_THRESH  | Spiking threshold                              | 1.0     |
| V_RESET   | Reset potential                                | 0.0     |
| TAU_0     | Parameter of the first_spike loss function     | 0.5     |
| TAU_1     | Parameter of the first_spike loss function     | 6.4     |
| ALPHA     | Parameter of the first_spike loss function     | 3e-3    |
| INPUT_HIDDEN_MEAN | Mean synaptic weight from input to hidden neurons | 1.5 |
| INPUT_HIDDEN_STD  | Standard deviation of synaptic weights from input to hidden neurons | 0.78 |
| HIDDEN_OUTPUT_MEAN | Mean synaptic weight from hidden rto output neurons | 0.93 |
| HIDDEN_OUTPUT_STD | Standard deviation of synaptic weights from hidden to output neurons | 0.1

### Learning parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| ETA       | Learning rate                                  | 5e-3    |
| ADAM_BETA1 | Adam optimiser parameter                      | 0.9     |
| ADAM_BETA2 | Adam optimiser parameter                      | 0.999   |
| ADAM_EPS   | Adam optimiser parameter                      | 1e-8    |
| ETA_DECAY  | Multiplier for learnign rate decay, applied every epoch | 0.95 |

### Recording controls
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| SPK_REC_STEPS | Size of the GeNN spike recording buffer in timesteps | TRIAL_MS/DT_MS |
| REC_SPIKES | List of neuron populations to record spikes from, possible entries "input", "hidden", "output" | [] |
|REC_NEURONS | List of pairs (neuron population, variable name) to record from, possible entries for population are "input", "hidden", "output" | [] |
|REC_SYNAPSES | List of pairs (synapse population, synapse variable name) to record from, possible entries for synapse population are "in_to_hid", "hid_to_out" | [] |
| WRITE_TO_DISK | Whether to write outputs to disk or just return them from the run function | True |
| TRAINING_PLOT | Whether to display plots during training | False |
| TRAINING_PLOT_INTERVAL | How often to display a training plot in number of epochs | 10 |
| FANCY_PLOTS | Whether to display additional plots | False |
| LOAD_LAST | Whether to load a checkpoint from a previous run | False |
|W_REPORT_INTERVAL | How often to save weight matrices; interval in number of trials | 100 |
| W_EPOCH_INTERVAL | How often to save weight matrices; interval in terms of epochs | 10 |


## MNIST parameters
Many parameters are the same across the different benchmarks but we list them againg for ease of reference.

### General parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| NAME      | A unique name for an experiment; results will be appended if a result file with this name already exists | "test" |
| DEBUG_HIDDEN_N | Whether to collect and return information about the activity levels of hidden neurons | False |
| OUT_DIR   | Directory where to write results               | "." (current dir) |
| DT_MS     | Time step od the simulations in millisceonds   | 1.0 |
| BUILD     | Whether to (re)build the GeNN model, can be set to False if there are repeated runs of the same model | True |
| TIMING   | Whether to record timing information during a run | True |


### Experiment parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| TRAIN_DATA_SEED | Seed for the random number generator used for generating training data | 123 |
| TEST_DATA_SEED | Seed for the random number generator used for generating testing data | 456 |
| MODEL_SEED | A separate random number seed for the random number generator that is used during model generation, e.g. for random initial values of synapse weights | None |
| TRIAL_MS  | Duration of trials in milliseconds             | 20.0    |
| N_MAX_SPIKE | Size of the bufferfor saved spikes in number of spikes; note the buffer contains saved spikes across two trials | 50 |
| N_TRAIN   | Number of training examples                    | 55000   |
| N_VALIDATE | Number of examples in the validation set      | 5000    |
| SHUFFLE   | Whether to shuffle the inputs in the training set | True |
| N_TEST    | Number of examples in the test set             | 10000   |

### Network structure
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| NUM_HIDDEN | Number of neurons in the hidden layer         | 128     |
| RECURRENT | Whether to include recurrent connections       | False   |

### Model parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| TAU_SYN   | Synaptic timescale in millisecons              | 5.0     |
| TAU_MEM   | Membrane times constant in milliseconds        | 20.0    |
| V_THRESH  | Spiking threshold                              | 1.0     |
| V_RESET   | Reset potential                                | 0.0     |
| INPUT_HIDDEN_MEAN | Mean synaptic weight from input to hidden neurons | 0.078 |
| INPUT_HIDDEN_STD | Standard deviation of synaptic weights from input to hidden neurons | 0.045 |
| HIDDEN_OUTPUT_MEAN | Mean synaptic weight from hidden rto output neurons | 0.2 |
| HIDDEN_OUTPUT_STD | Standard deviation of synaptic weights from hidden to output neurons | 0.37 |
| HIDDEN_HIDDEN_MEAN | Mean of initial synaptic weights of hidden to hidden recurrent connections | 0.2 |
| HIDDEN_HIDDEN_STD | Mean of initial synaptic weights of hidden to hidden recurrent connections | 0.37 | 
| PDROP_INPUT | Probability of dropping input spikes | 0.2 |
| PDROP_HIDDEN | Probability of dropping spikes in the hidden layer | 0.0 |
| REG_TYPE | Type of regularisation to apply to the hidden layer | "none" |
| LBD_UPPER | Regularisation strength of per-neuron regularisation for hyper-active neurons | 0.000005 |
| LBD_LOWER | Regularisation strength of per-neuron regularisation for neurons with low activity | 0.001 |
| NU_UPPER | Target activity level per hidden neuron per trial"]= 2 |
| RHO_UPPER | Target activity level for the entire hidden layer per trial | 5000.0 |
| GLB_UPPER | Strength of regularisation based on global number of spikes per trial (type "Thomas1" | 1e-5 |

### Learning parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| ETA       | Learning rate                                  | 5e-3    |
| ADAM_BETA1 | Adam optimiser parameter                      | 0.9     |
| ADAM_BETA2 | Adam optimiser parameter                      | 0.999   |
| ADAM_EPS   | Adam optimiser parameter                      | 1e-8    |

### Recording controls
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| W_OUTPUT_EPOCH_TRIAL | List of 2-entry lists [epoch,trial] at wich to save weights (replaes the intrvals above) | [] |
| SPK_REC_STEPS | Size of the GeNN spike recording buffer in timesteps | TRIAL_MS/DT_MS |
| REC_SPIKES_EPOCH_TRIAL | Controls at which [epoch,trial] to record spikes | [] |
| REC_SPIKES | List of neuron populations to record spikes from, possible entries "input", "hidden", "output" | [] |
| REC_NEURONS_EPOCH_TRIAL | Controls at which [epoch,trial] to record neuron vars | [] |
|REC_NEURONS | List of pairs (neuron population, variable name) to record from, possible entries for population are "input", "hidden", "output" | [] |
| REC_SYNAPSES_EPOCH_TRIAL | Controls at which [epoch,trial] to record synapse vars | [] |
|REC_SYNAPSES | List of pairs (synapse population, synapse variable name) to record from, possible entries for synapse population are "in_to_hid", "hid_to_out" | [] |
| WRITE_TO_DISK | Whether to write outputs to disk or just return them from the run function | True |
| LOAD_LAST | Whether to load a checkpoint from a previous run | False |

### Loss types and parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| LOSS_TYPE | The loss function to use, possible values "first_spike", "first_spike_exp", "max", "sum", "avg_xentropy" | "avg_xentropy" |
| EVALUATION | How to form a validation set | "random" |
| CUDA_VISIBLE_DEVICES | Internal GeNN switch how CUDA devices are addressed | True |
| AVG_SNSUM | Whether to average spike counts across a mini-batch for regularisation spike counts | False |
| REDUCED_CLASSES | A list of classes to train; if None, all classes are trained | None |
| TAU_0 | Parameter of the first_spike loss functions | 0.5 |
| TAU_1 | Parameter of the first_spike loss functions | 6.4 |
| ALPHA | Parameter of the first_spike loss functions | 3e-3 |



## SHD/SSC parameters

### General parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| NAME      | A unique name for an experiment; results will be appended if a result file with this name already exists | "test" |
| DEBUG_HIDDEN_N | Whether to collect and return information about the activity levels of hidden neurons | False |
| OUT_DIR   | Directory where to write results               | "." (current dir) |
| DT_MS     | Time step od the simulations in millisceonds   | 1.0 |
| BUILD     | Whether to (re)build the GeNN model, can be set to False if there are repeated runs of the same model | True |
| TIMING   | Whether to record timing information during a run | True |


### Experiment parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| TRAIN_DATA_SEED | Seed for the random number generator used for generating training data | 123 |
| TEST_DATA_SEED | Seed for the random number generator used for generating testing data | 456 |
| MODEL_SEED | A separate random number seed for the random number generator that is used during model generation, e.g. for random initial values of synapse weights | None |
| TRIAL_MS  | Duration of trials in milliseconds             | 1400.0 |
| N_MAX_SPIKE | Size of the bufferfor saved spikes in number of spikes; note the buffer contains saved spikes across two trials | 1500 |
| N_TRAIN   | Number of training examples                    | 7644  |
| N_VALIDATE | Number of examples in the validation set      | 512    |
| SHUFFLE   | Whether to shuffle the inputs in the training set | True |

### Network structure
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| NUM_HIDDEN | Number of neurons in the hidden layer         | 256     |
| RECURRENT | Whether to include recurrent connections       | False   |
| N_HID_LAYER | Number of hidden layers                      | 1       |

### Model parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| TAU_SYN   | Synaptic timescale in millisecons              | 5.0     |
| TAU_MEM   | Membrane times constant in milliseconds        | 20.0    |
| V_THRESH  | Spiking threshold                              | 1.0     |
| V_RESET   | Reset potential                                | 0.0     |
| HIDDEN_NEURON_TYPE | Hidden neuron type, possible values "LIF", "hetLIF", ALIF" | "LIF" |
| OUTPUT_NEURON_TYPE | Output neuron type, possible values "LI", "hetLI" | "LI" |
| TAU_B    | Timascale of adaptation of ALIF neurons         | 100.0   |
| B_INCR | Increment of the spike rate adaptation variable B upon a spike | 0.1 |
| B_INIT | Initial value of the spike rate adaptation variable B | 0.0 |
| INPUT_HIDDEN_MEAN | Mean synaptic weight from input to hidden neurons | 0.078 |
| INPUT_HIDDEN_STD | Standard deviation of synaptic weights from input to hidden neurons | 0.045 |
| HIDDEN_OUTPUT_MEAN | Mean synaptic weight from hidden rto output neurons | 0.2 |
| HIDDEN_OUTPUT_STD | Standard deviation of synaptic weights from hidden to output neurons | 0.37 |
| HIDDEN_HIDDEN_MEAN | Mean of initial synaptic weights of hidden to hidden recurrent connections | 0.2 |
| HIDDEN_HIDDEN_STD | Mean of initial synaptic weights of hidden to hidden recurrent connections | 0.37 | 
| HIDDEN_HIDDENFWD_MEAN | Mean synaptic weight of forward synapses between multiple hidden layers | 0.02  |
| HIDDEN_HIDDENFWD_STD | Standard deviation of weights of forward synapses between multiple hidden layers | 0.01 |
| PDROP_INPUT | Probability of dropping input spikes | 0.2 |
| PDROP_HIDDEN | Probability of dropping spikes in the hidden layer | 0.0 |
| REG_TYPE | Type of regularisation to apply to the hidden layer | "none" |
| LBD_UPPER | Regularisation strength of per-neuron regularisation for hyper-active neurons | 0.000005 |
| LBD_LOWER | Regularisation strength of per-neuron regularisation for neurons with low activity | 0.001 |
| NU_UPPER | Target activity level per hidden neuron per trial"]= 2 |
| RHO_UPPER | Target activity level for the entire hidden layer per trial | 5000.0 |
| GLB_UPPER | Strength of regularisation based on global number of spikes per trial (type "Thomas1" | 1e-5 |
| REWIRE_SILENT | Whether to rewire silent hidden neurons | False |
| REWIRE_LIFT | Whether and how much to uplift incoming synapse weights of silent hidden neurons (if not zero, this replaces rewiring) | 0.0 |

### Learning parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| ETA       | Learning rate                                  | 5e-3    |
| ADAM_BETA1 | Adam optimiser parameter                      | 0.9     |
| ADAM_BETA2 | Adam optimiser parameter                      | 0.999   |
| ADAM_EPS   | Adam optimiser parameter                      | 1e-8    |
| EMA_ALPHA1 | Factor of the fast exponential moving average of accuracy (used for LR schedule) | 0.8 |
| EMA_ALPHA2 |  Factor of the fast exponential moving average of accuracy (used for LR schedule) | 0.95 |
| ETA_FAC | Factor by which to multiply the learning rate when applying the LR schedule | 0.5 |
| MIN_EPOCH_ETA_FIXED | Minimal number of epochs that the learning rate stays fixed during a learning rate schedule | 300 |
| TRAIN_TAU | Whether to train tau values | False |
| MIN_TAU_M | Lower bound for the value of tau_m | 3.0 |
| MIN_TAU_SYN | Lower bound for the value of tau_syn | 1.0 |
| TRAIN_TAU_OUTPUT | Whether to train the timescale of the output neurons | False |
| TRAIN_W | Whether to train synaptic weights | True |
| TRAIN_W_OUTPUT | Whether to train the synaptic weights towards the output neurons | True |

### Recording controls
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| W_OUTPUT_EPOCH_TRIAL | List of 2-entry lists [epoch,trial] at wich to save weights  | [] |
| TAU_OUTPUT_EPOCH_TRIAL | List of 2-entry lists [epoch,trial] at wich to save tau values | [] |
| SPK_REC_STEPS | Size of the GeNN spike recording buffer in timesteps | TRIAL_MS/DT_MS |
| REC_SPIKES_EPOCH_TRIAL | Controls at which [epoch,trial] to record spikes | [] |
| REC_SPIKES | List of neuron populations to record spikes from, possible entries "input", "hidden", "output" | [] |
| REC_NEURONS_EPOCH_TRIAL | Controls at which [epoch,trial] to record neuron vars | [] |
|REC_NEURONS | List of pairs (neuron population, variable name) to record from, possible entries for population are "input", "hidden", "output" | [] |
| REC_SYNAPSES_EPOCH_TRIAL | Controls at which [epoch,trial] to record synapse vars | [] |
|REC_SYNAPSES | List of pairs (synapse population, synapse variable name) to record from, possible entries for synapse population are "in_to_hid", "hid_to_out" | [] |
| WRITE_TO_DISK | Whether to write outputs to disk or just return them from the run function | True |
| LOAD_LAST | Whether to load a checkpoint from a previous run | False |
| LOAD_BEST | Whether to load a checkpoint of a best observed training stage | False |

### Loss types and parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| LOSS_TYPE | The loss function to use, possible values "first_spike", "first_spike_exp", "max", "sum", "sum_weigh_linear", sum_weigh_exp", "sum_weigh_sigmoid", "sum_weigh_input", "avg_xentropy" | "sum_weigh_exp" |
| EVALUATION | How to form a validation set, values "random" or "speaker" | "random" |
| SPEAKER_LEFT | List of speakers to leave out (ordinal number in the list of speakers not speaker IDs - values 0-9 allowed) | [0] |
| CUDA_VISIBLE_DEVICES | Internal GeNN switch how CUDA devices are addressed | True |
| AVG_SNSUM | Whether to average spike counts across a mini-batch for regularisation spike counts | False |
| REDUCED_CLASSES | A list of classes to train; if None, all classes are trained | None |
| TAU_0 | Parameter of the first_spike loss functions | 1.0 |
| TAU_1 | Parameter of the first_spike loss functions | 100.0 |
| ALPHA | Parameter of the first_spike loss functions | 5e-5 |
| AUGMENTATION | Dictionary of augmentations to apply to the training data, possible values {"random_shift": x}, {"random_dilate": [y0, y1]}, {"ID_jitter": z} where x is max size of shoft, y0,y1 min/max dilation factors, z range of the jitter across input channels | {} |
| COLLECT_CONFUSION | Whether to collect confusion matrices during training / testing | False |
| REC_PREDICTIONS | Whether to record the individual predictions during training / testing | False |
