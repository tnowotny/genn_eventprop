# genn_eventprop
Implementation of eventprop learning in GeNN


The repository contains research code that was used to make a first implementation of the Eventprop learning rule [CITE] and test it against increasingly difficult benchmarks. The code underlies the publication: Nowotny, T., Turner, J.P. and Knight, J.C., 2022. Loss shaping enhances exact gradient learning with eventprop in spiking neural networks. arXiv preprint arXiv:2212.01232.

The code is generally organised so that the GeNN model definitions are colected in `models.py` and for the benchmarks there are individual python files `simulator_XXX.py` as follows:

- `simulator_yinyang.py`: Train to classify the Yinyang benchmark dataset [CITE]
- `simulator_MNIST.py`: Train to classify the MNIST dataset [CITE]
- `simulator_SHD.py`: Train to classify the Spikeing Heidelberg Digits (SHD) [CITE] or Spiking Google Speech Commands (SSC) [CITE] datasets. Which to train is selected with parameter `p["DATASET"]`
- `simulator_DVS_gesture.py`: *experimental*, train to classify the DVS gesture dataset [CITE]
- `simulator_SMNIST.py`: *experimental*, train to classify the sequential MNIST dataset [CITE]

The latter two simulators are in an early experimental stage and did not contribute to the publication.

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
| TAU_0     |
| TAU_1     |
| ALPHA     |
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
Many parameters are the same across the different benchmarks. The following tables lists only the ones that are different for MNIST compared to YinYang.

### General parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| NAME      | A unique name for an experiment; results will be appended if a result file with this name already exists | "test" |
| DEBUG_HIDDEN_N | Whether to collect and return information about the activity levels of hidden neurons | False |
| DT_MS | as above | 1.0 |


### Experiment parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| MODEL_SEED | A separate random number seed for the random number generator that is used during model generation, e.g. for random initial values of synapse weights | None |
| TRIAL_MS  | Duration of trials in milliseconds             | 20.0    |
| N_MAX_SPIKE | as above | 50 |
| N_TRAIN   | Number of training examples                    | 55000   |
| N_VALIDATE | Number of examples in the validation set      | 5000    |
| SHUFFLE   | Whether to shuffle the inputs in the training set | True |
| N_TEST    | Number of examples in the test set             | 10000   |

### Network structure
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| NUM_HIDDEN | Number of neurons in the hidden layer         | 128     |
| RECURRENT | Whether to include recurrent connections       | False   |
| INPUT_HIDDEN_MEAN | As above | 0.078 |
| INPUT_HIDDEN_STD | As above  | 0.045 |
| HIDDEN_OUTPUT_MEAN | As above  | 0.2 |
| HIDDEN_OUTPUT_STD | As above | 0.37 |
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

### Recording controls
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
W_OUTPUT_EPOCH_TRIAL | List of 2-entry lists [epoch,trial] at wich to save weights (replaes the intrvals above) | [] |
| REC_SPIKES_EPOCH_TRIAL | Controls at which [epoch,trial] to record spikes | [] |
| REC_NEURONS_EPOCH_TRIAL | Controls at which [epoch,trial] to record neuron vars | [] |
| REC_SYNAPSES_EPOCH_TRIAL | Controls at which [epoch,trial] to record synapse vars | [] |

### Loss types and parameters
| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| LOSS_TYPE | The loss function to use, possible values "first_spike", "first_spike_exp", "max", "sum", "avg_xentropy" | "avg_xentropy" |
| EVALUATION | How to form a validation set | "random" |
| CUDA_VISIBLE_DEVICES | Internal GeNN switch how CUDA devices are addressed | True |
| AVG_SNSUM | Whether to average spike counts across a mini-batch for regularisation spike counts | False |
| REDUCED_CLASSES | A list of classes to train; if None, all classes are trained | None |
TAU_0 | Parameter of the first_spike loss functions | 0.5 |
TAU_1 | Parameter of the first_spike loss functions | 6.4 |
ALPHA | Parameter of the first_spike loss functions | 3e-3 |

## SHD/SSC parameters


| Name      | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| AUGMENTATION | Dictionary of augmentations to apply to the training data, possible values {"random_shift": x}, {"random_dilate": [y0, y1]}, {"ID_jitter": z} where x is max size of shoft, y0,y1 min/max dilation factors, z range of the jitter across input channels | {} |
