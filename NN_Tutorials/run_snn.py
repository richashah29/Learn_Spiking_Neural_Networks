import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate  #for approximating gradients
from snntorch import utils  #for things like resetting hidden neuron states

num_steps = 25  #simulate 25 time steps (temporal dim)
batch_sz = 1
beta = 0.5  # neuron leak/decay rate (memory of previous input that remains)
spike_grad = surrogate.fast_sigmoid() #surrogate gradient; need as spikes not differentiable

net = nn.Sequential(
      nn.Conv2d(1, 8, 5), ## extracts spatial features from 28x28 img
      nn.MaxPool2d(2),   ## downsample spatial dimensions
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad), ## LIF neuron layer; integrates input over time, spikes if poten. exceeds threshold, decays over time by beta

      ## init_hidden=True stores neuron states (membrane potential) across time

      nn.Conv2d(8, 16, 5),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
      nn.Flatten(),
      nn.Linear(16 * 4 * 4, 10),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True))

      ## output=True; last layer marked output layer to help in training logic

data_in = torch.rand(num_steps, batch_sz, 1, 28, 28)  #random input data
## mimics something like MNIST images streamed over time
spike_recording = [] # record spikes over time
utils.reset(net) # reset/initialize hidden states for all neurons

for step in range(num_steps): # loop over time
    spike, state = net(data_in[step]) # one time step of forward-pass; each spiking layer processes input & returns the spike and state
    spike_recording.append(spike) # record spikes in list


## ^ time based simulation of a spiking CNN
## each layer has LIF neurons with custom decay and uses surrogate gradients