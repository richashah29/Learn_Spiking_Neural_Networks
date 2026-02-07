import snntorch as snn
import torch

# training parameters
batch_sz = 128
data_path = '/tmp/data/mnist'
num_classes = 10 ## MNIST has 10 output classes

# Torch Variables
dtype = torch.float


## Download Dataset
from torchvision import datasets, transforms

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

# # temporary dataloader if MNIST service is unavailable
# !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
# !tar -zxvf MNIST.tar.gz

# mnist_train = datasets.MNIST(root = './', train=True, download=True, transform=transform)

from snntorch import utils

subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of mnist_train is {len(mnist_train)}")

## Create Dataloaders
from torch.utils.data import DataLoader
train_loader = DataLoader(mnist_train, batch_size=batch_sz, shuffle=True)


### Spike Encoding

# MNIST is not a time-varying dataset so we can either:
# pass same training sample to network each time step (convert into static, unchanging video; each value of mxn X matrix is high precision value normalized between 0 and 1)

# convert input into spike train of sequence length num_steps where each feature/pixel has discrete value (converts into time-varying sequence of spikes featuring a relation to original image)

## 3 ways for spike encoding: rate coding (input features determine spiking frequence), latency coding (input features to determine spike timing), delta modulation (temporal change of input features to generate spikes)

## Rate Coding of MNIST
# each normalized input feature (Xij) used as prob. event occurs at any time (returns rate-coded value Rij) <-- Bernoulli Trail

# Temporal Dynamics
num_steps = 10

# create vector filled with 0.5
raw_vector = torch.ones(num_steps) * 0.5

# pass each sample through a Bernoulli trial
rate_coded_vector = torch.bernoulli(raw_vector)

print(f'Converted vector: {rate_coded_vector}')
print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")

## trying with longer vector (as n->inf, proportion of spikes apporach og raw value):
num_steps = 100
raw_vector = torch.ones(num_steps)*0.5
rate_coded_vector = torch.bernoulli(raw_vector)
print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")

## spikegen.rate converts images into spike trains
from snntorch import spikegen

# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)  #fetches a batch of label data
## data_it of shape (128, 1, 28, 28); one batch of 128 grayscale images
## targets_it is corresponding labels for those 128 images

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)

print(spike_data.size())
torch.Size([100, 128, 1, 28, 28])

## Visualization
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt # simplifies process of visualization
from IPython.display import HTML

# [T x B x 1 x 28 x 28] spike_data dim

spike_data_sample = spike_data[:,0,0] # one sample
print(spike_data_sample.size())

fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'

HTML(anim.to_html5_video())

# If you're feeling sentimental, you can save the animation: .gif, .mp4 etc.
# anim.save("spike_mnist_test.mp4")

print(f"The corresponding target is: {targets_it[0]}")

# reduce spiking frequency to 25%
spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)

spike_data_sample2 = spike_data[:,0,0]
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample2, fig, ax)
HTML(anim.to_html5_video())

plt.figure(facecolor="w")
plt.subplot(1,2,1)
plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain=1')

plt.subplot(1,2,2)
plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain=0.25')

plt.show()

# when gain=0.25, the image becomes lighter as spiking probability is reduced by a factor of 4

## Raster Plots - show input; replot into 2D tensor

# Reshape
spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))

# raster plot
fig = plt.figure(facecolor="w", figsize=(10,5))
ax = fig.add_subplot(111)
splt.raster(spike_data_sample2, ax, s=1.5, c="black")

plt.title("Input Layer")
plt.xlabel("Time Step")
plt.ylabel("Neuron Number")
plt.show()

# index into one neutron
idx = 400  #anywhere between 0-784

fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

splt.raster(spike_data_sample.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.show()

### Latency coding of MNIST
# temperal codes capture info of firing time of neuron (single spike has more meaning than in rate codes which rely on firing freq.) --> susceptible to noise but less power required

# spikegen.latency is a func that allows each input to fire at most once during full time sweep (features closer to 1 spike earlier; in MNIST, brighter pixels fire earlier)

def convert_to_time(data, tau=5, threshold=0.01):
    spike_time = tau* torch.log(data/(data-threshold))
    return spike_time

# realtionship between input feature intensity & spike time
raw_input = torch.arrange(0, 5, 0.05)  #tensor from 0 to 5
spike_times = convert_to_time(raw_input) # times when spikes triggered

plt.plot(raw_input, spike_times)
plt.xlabel('Input Value')
plt.ylabel('Spike Time (s)')
plt.show()

# the smaller the value the later the spike occurs (exp. dependence)
spike_data = spikegen.latency(data_it, num_steps=100, threshold=0.01)

# arguments: tau, RC cct; higher tau slower firing + threshold, the membrane potential firing threshold (input values below clipped and assigned to final time step)

# raster plot for data:
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# optional save
# fig.savefig('destination_path.png', format='png', dpi=300)

## ^ due to high contrast and lack of greyscale features, there's clustering in 2 areas of the plot; can increase tau to slow down firing or set optional argument: linear=True

# example:
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, linear=True)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# all firing in ~5 time steps; rest do nothing so very redundant
# increase tau or set optional argument normalize=True to span full num_steps

spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, normalize=True, linear=True)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

## major advantage of latency coding over rate coding is sparsity (if neurons constrained to firing a max of once over time course of interest then it promotes low-power operation)

## remove redundant features (the spikes that occur at final time step) by setting clip=True

# eg.
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, clip=True, normalize=True, linear=True)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()


## animation:
spike_data_sample = spike_data[:, 0, 0]
print(spike_data_sample.size()) #torch.Size([100, 28, 28])

fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
HTML(anim.to_html5_video())

# Save output: .gif, .mp4 etc.
# anim.save("mnist_latency.gif")
print(targets_it[0])

### Delta Modulation
# biology is event-driven; neurons thrive on change
# based on event-driven spiking; time-series tensor input takes difference between subsequent feature across all time steps and if difference > Vth and 0 then spike

# Create a tensor with some fake time-series data
data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])

# Plot the tensor
plt.plot(data)

plt.title("Some fake time-series data")
plt.xlabel("Time step")
plt.ylabel("Voltage (mV)")
plt.show()

# Convert data
spike_data = spikegen.delta(data, threshold=4) # arbitary threshold

# Create fig, ax
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

# Raster plot of delta converted data
splt.raster(spike_data, ax, c="black")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()

# 3 time steps where difference between data[T] and data[T+1] > Vth (thus 3 on-spikes)
## if negative swings are important, enable off_spike=True

# Convert data
spike_data = spikegen.delta(data, threshold=4, off_spike=True)

# Create fig, ax
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

# Raster plot of delta converted data
splt.raster(spike_data, ax, c="black")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()

# off-spikes take on value of -1
print(spike_data) # tensor([0., 0., 0., 1., -1., ])

## true use of spikegen.delta: compress time-series data by only generating spikes for sufficiently large changes/events

### Spike Generation

# Create a random spike train
spike_prob = torch.rand((num_steps, 28, 28), dtype=dtype) * 0.5
spike_rand = spikegen.rate_conv(spike_prob)

fig, ax = plt.subplots()
anim = splt.animator(spike_rand, fig, ax)
HTML(anim.to_html5_video())

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_rand[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()


