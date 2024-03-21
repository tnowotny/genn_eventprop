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
| NUM_INPUT

## MNIST parameters

## SHD/SSC parameters


