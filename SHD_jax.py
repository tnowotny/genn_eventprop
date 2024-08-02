import os

import numpy as np

import jax
import jax.numpy as jnp
import spyx
import haiku as hk
import optax
from jax_tqdm import scan_tqdm


import tonic
from argparse import ArgumentParser
from tonic import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import namedtuple


#jax.config.update("jax_default_matmul_precision", "float32")

State = namedtuple("State", "obs labels")

class _SHD2Raster():
    """ 
    Tool for rastering SHD samples into frames. Packs bits along the temporal axis for memory efficiency. This means
        that the used will have to apply jnp.unpackbits(events, axis=<time axis>) prior to feeding the data to the network.
    """

    def __init__(self, encoding_dim, sample_T = 100):
        self.encoding_dim = encoding_dim
        self.sample_T = sample_T
        
    def __call__(self, events):
        # tensor has dimensions (time_steps, encoding_dim)
        tensor = np.zeros((events["t"].max()+1, self.encoding_dim), dtype=int)
        np.add.at(tensor, (events["t"], events["x"]), 1)
        #return tensor[:self.sample_T,:]
        tensor = tensor[:self.sample_T,:]
        tensor = np.minimum(tensor, 1)
        tensor = np.packbits(tensor, axis=0)
        return tensor

parser = ArgumentParser(description="JAX RSNN")
# compilation options
parser.add_argument("--sample-t", type=int, default=200)
parser.add_argument("--dt", type=float, default=5.0)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-hidden", type=int, default=256)
parser.add_argument("--num-trials", type=int, default=1)
parser.add_argument("--num-epochs", type=int, default=100)

args = parser.parse_args()

shd_timestep = 1e-6
shd_channels = 700
net_channels = 700
net_dt = args.dt * 1e-3

obs_shape = tuple([net_channels,])
act_shape = tuple([20,])

transform = transforms.Compose([
    transforms.Downsample(
        time_factor=shd_timestep / net_dt,
        spatial_factor=net_channels / shd_channels
    ),
    _SHD2Raster(net_channels, sample_T=args.sample_t)
])

train_dataset = datasets.SHD("./data", train=True, transform=transform)
test_dataset = datasets.SHD("./data", train=False, transform=transform)

train_dl = iter(DataLoader(train_dataset, batch_size=len(train_dataset),
                           collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))

x_train, y_train = next(train_dl)

test_dl = iter(DataLoader(test_dataset, batch_size=len(test_dataset),
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))

x_test, y_test = next(test_dl)

x_train = jnp.array(x_train, dtype=jnp.uint8)
y_train = jnp.array(y_train, dtype=jnp.uint8)
print(f"{(x_train.nbytes + y_train.nbytes) / (1024 * 1024)} MiB training data")

def _shuffle(dataset, shuffle_rng, batch_size):
    x, y = dataset

    full_batches = y.shape[0] // batch_size

    indices = jax.random.permutation(shuffle_rng, y.shape[0])[:full_batches*batch_size]
    obs, labels = x[indices], y[indices]

    obs = jnp.reshape(obs, (-1, batch_size) + obs.shape[1:])
    labels = jnp.reshape(labels, (-1, batch_size)) # should make batch size a global

    return obs, labels

shuffle = jax.jit(_shuffle, static_argnums=2)
def build_snn(hidden_shape, batch_size):

    def shd_snn(x): 
        # **NOTE** beta = membrane
        core = hk.DeepRNN([
            hk.Linear(hidden_shape, with_bias=False),
            spyx.nn.RCuBaLIF((hidden_shape,), alpha=np.exp(-net_dt/5e-3), beta=np.exp(-net_dt/20e-3),
                             activation=spyx.axn.arctan()),
            hk.Linear(20, with_bias=False),
            spyx.nn.LI((20,), beta=np.exp(-net_dt/20e-3))
        ])
        
        # static unroll for maximum performance
        spikes, V = hk.static_unroll(core, x, core.initial_state(x.shape[0]), time_major=False)
        
        return spikes, V
    
    key = jax.random.PRNGKey(0)
    # Since there's nothing stochastic about the network, we can avoid using an RNG as a param!
    sample_x, sample_y = shuffle((x_train,y_train),key, batch_size)
    SNN = hk.without_apply_rng(hk.transform(shd_snn))
    params = SNN.init(rng=key, x=jnp.float32(sample_x[0]))
    
    return SNN, params

def benchmark(SNN, params, dataset, epochs, batch_size):
        
    opt = optax.adam(learning_rate=5e-4)
    # create and initialize the optimizer
    opt_state = opt.init(params)
    grad_params = params

    Loss = spyx.fn.integral_crossentropy()
    # define and compile our eval function that computes the loss for our SNN
    @jax.jit
    def net_eval(weights, events, targets):
        readout = SNN.apply(weights, events)
        traces, V_f = readout
        return Loss(traces, targets) # smoothing needs to be more explicit in docs...
        
    # Use JAX to create a function that calculates the loss and the gradient!
    surrogate_grad = jax.value_and_grad(net_eval) 
    rng = jax.random.PRNGKey(0)        
    
    # compile the meat of our training loop for speed
    @jax.jit
    def train_step(state, data):
        grad_params, opt_state = state
        events, targets = data
        # events = jnp.swapaxes(events, 0, 1)
        events = jnp.unpackbits(events, axis=1, count=args.sample_t) # decompress temporal axis
        # compute loss and gradient                    # need better augment rng
        loss, grads = surrogate_grad(grad_params, events, targets)
        # generate updates based on the gradients and optimizer
        updates, opt_state = opt.update(grads, opt_state, grad_params)
        # return the updated parameters
        new_state = [optax.apply_updates(grad_params, updates), opt_state]
        return new_state, loss
    
    
    # Here's the start of our training loop!
    @scan_tqdm(epochs)
    def epoch(epoch_state, epoch_num):
        curr_params, curr_opt_state = epoch_state

        shuffle_rng = jax.random.fold_in(rng, epoch_num)
        train_data = shuffle(dataset, shuffle_rng, batch_size)
        
        # train epoch
        end_state, train_loss = jax.lax.scan(
            train_step,# func
            [curr_params, curr_opt_state],# init
            train_data,# xs
        )
                    
        return end_state, jnp.mean(train_loss)
    # end epoch
    
    # epoch loop
    final_state, metrics = jax.lax.scan(
        epoch,
        [grad_params, opt_state], # metric arrays
        jnp.arange(epochs), # 
        epochs # len of loop
    )
    
    final_params, _ = final_state
    
                
    # return our final, optimized network.       
    return final_params, metrics

from time import time


def run_bench(trials, num_epochs, net_width, batch_size):
    print(f"{net_width} hidden neurons, {batch_size} batch size")
    SNN, params = build_snn(net_width, batch_size)
    
    times = []
    for t in range(trials+1):
        print(t, ":", end="")
        start = time()
        benchmark(SNN, params, (x_train,y_train), num_epochs, batch_size)
        times.append(time() - start)
        print(times[t])
    
    print("\tMean:", np.mean(times[1:]), "Std. Dev.:", np.std(times[1:]))

run_bench(args.num_trials, args.num_epochs, args.num_hidden, args.batch_size)

