# This code aims to create a simple spiking neural network encoder for processing video frames inspired by the SAD model.

# apply 2d convolution layer <-- kernel=3, stride=1, padding=1
# batch normalization (scale down, 1D)
# apply spiking neuron layer --> LIF neurons; instead of weights like in regular CNNs
    
import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron

class SimpleSTMEncoder(nn.Module):
    def __init__(self, in_channels=3, T=4, feature_dim=64, depth_dim=48):
        super().__init__()
        self.T = T

        # Encoder: 12-layer inverted bottleneck structure
        channels = [in_channels, 64, 128, 256, 384, 512, 512, 384, 256, 128, 64, 64, 64]
        self.encoder = nn.ModuleList()

        for i in range(12):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                neuron.MultiStepLIFNode(tau=2.0, detach_reset=True)
            ))

        # Output heads
        self.feature_head = nn.Conv2d(channels[-1], feature_dim, kernel_size=1)
        self.depth_head = nn.Conv2d(channels[-1], depth_dim, kernel_size=1)

    def forward(self, x):
        # x shape: [B, C, T, H, W]
        x = x.permute(2, 0, 1, 3, 4).contiguous()  # [T, B, C, H, W]

        for block in self.encoder:
            conv, bn, lif = block[0], block[1], block[2]
            x = x.view(self.T * x.size(1), x.size(2), x.size(3), x.size(4))
            x = conv(x)
            x = bn(x)
            x = x.view(self.T, -1, x.size(1), x.size(2), x.size(3))
            x = lif(x)

        # Output: average spike rate
        x = x.permute(1, 2, 0, 3, 4)  # [B, C, T, H, W]
        x_avg = torch.mean(x, dim=2)  # firing rate

        # Heads
        F = self.feature_head(x_avg)  # [B, Cf, H, W]
        D = self.depth_head(x_avg)    # [B, Cd, H, W]
        return F.unsqueeze(2), D.unsqueeze(2)


# Test example
if __name__ == "__main__":
    encoder = SimpleSTMEncoder()
    x = torch.randn(2, 3, 4, 64, 64)  # [B, C, T, H, W]
    F, D = encoder(x)

    print("Feature map F:", F.shape)  # [B, Cf, L, H, W]
    print("Depth map D:  ", D.shape)  # [B, Cd, L, H, W]

    # Frustum outer product: [B, Cf, Cd, L, H, W]
    frustum = F.unsqueeze(2) * D.unsqueeze(1)
    print("Frustum shape:", frustum.shape)

    print('Frustum outer product shape:', frustum.shape)  # [B, Cf, Cd, L, H, W]
    