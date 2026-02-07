'''
[Repeated Image Input]
    ↓
[Conv2d → LIF] x 2 layers
    ↓
[Spikes over T timesteps]
'''

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np

class NuScenesCameraDataset(Dataset):
    def __init__(self, nusc, split='v1.0-mini', camera='CAM_FRONT', T=5):
        self.nusc = nusc
        self.T = T
        self.camera = camera
        self.samples = [s for s in nusc.sample]

        self.transform = transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples) - self.T

    def __getitem__(self, idx):
        frames = []
        for t in range(self.T):
            sample = self.samples[idx + t]
            cam_token = sample['data'][self.camera]
            cam_data = self.nusc.get('sample_data', cam_token)
            img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
            img = Image.open(img_path).convert('RGB')
            frames.append(self.transform(img))
        x = torch.stack(frames)  # [T, C, H, W]

        # Zero tensor for labels (not used in this example)
        y = torch.zeros((128, 256), dtype=torch.long)
        return x, y

class SimpleSTM(nn.Module):
    def __init__(self, T=5, beta=0.9):
        super().__init__()
        self.T = T

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):  # x: [T, C, H, W]
        mem1, mem2 = None, None
        spike_seq1, spike_seq2 = [], []

        for t in range(self.T):
            out = self.conv1(x[t])
            mem1, spk1 = self.lif1(out, mem1)
            spike_seq1.append(spk1)

            out = self.conv2(spk1)
            mem2, spk2 = self.lif2(out, mem2)
            spike_seq2.append(spk2)

        return torch.stack(spike_seq1), torch.stack(spike_seq2)  # [T, C, H, W]

T = 5
stm = SimpleSTM(T=T)
nusc = NuScenes(version='v1.0-mini', dataroot='path/to/NuScenes_images', verbose=True)
dataset = NuScenesCameraDataset(nusc, camera='CAM_FRONT', T=5)
x, y = dataset[0]
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
spk1, spk2 = stm(x)

fig1, ax1 = plt.subplots()
splt.raster(spk1[:, 0, 32, 32].unsqueeze(0), ax=ax1)
ax1.set_title("Layer 1 Spike Raster")

fig2, ax2 = plt.subplots()
splt.raster(spk2[:, 0, 32, 32].unsqueeze(0), ax=ax2)
ax2.set_title("Layer 2 Spike Raster")

plt.show()
