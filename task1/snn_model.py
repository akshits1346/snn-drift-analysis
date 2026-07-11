"""
Shared SNN model definition.

Architecture: Small VGG-style SNN (3 conv blocks + 1 FC).
  - Input: [B, T, C, H, W]  (batch, timesteps, channels, height, width)
  - Output: [B, T, n_classes]

LIF neuron uses the ZIF (triangle) surrogate gradient from TET.
This model is used by BOTH training algorithms so results are comparable.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Surrogate gradient (triangle / ZIF, from TET paper)
# ---------------------------------------------------------------------------
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input > 0).float()
        ctx.save_for_backward(input, torch.tensor([gamma]))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma_t = ctx.saved_tensors
        gamma = gamma_t.item()
        tmp = (1 / gamma) ** 2 * ((gamma - input.abs()).clamp(min=0))
        return grad_output * tmp, None


# ---------------------------------------------------------------------------
# LIF neuron layer
# ---------------------------------------------------------------------------
class LIFSpike(nn.Module):
    """
    Leaky Integrate-and-Fire with hard reset.
    Membrane potential decays with tau (default 0.5).
    Fires when mem >= thresh, then resets to 0.
    """
    def __init__(self, thresh=1.0, tau=0.5, gamma=1.0):
        super().__init__()
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma

    def forward(self, x):
        # x: [B, T, ...]
        mem = torch.zeros_like(x[:, 0])
        spikes = []
        for t in range(x.shape[1]):
            mem = mem * self.tau + x[:, t]
            spike = ZIF.apply(mem - self.thresh, self.gamma)
            mem = (1 - spike) * mem          # hard reset
            spikes.append(spike)
        return torch.stack(spikes, dim=1)    # [B, T, ...]


# ---------------------------------------------------------------------------
# Helper: apply an ANN-style module across the time dimension
# ---------------------------------------------------------------------------
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T = x.shape[:2]
        out = self.module(x.flatten(0, 1))   # [B*T, ...]
        return out.view(B, T, *out.shape[1:])


# ---------------------------------------------------------------------------
# Conv + BN + LIF block
# ---------------------------------------------------------------------------
class ConvLIFBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv_bn = TimeDistributed(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.lif = LIFSpike()

    def forward(self, x):
        return self.lif(self.conv_bn(x))


# ---------------------------------------------------------------------------
# Small SNN: 3 conv blocks, global-avg-pool, 1 FC
# ---------------------------------------------------------------------------
class SmallSNN(nn.Module):
    """
    Input:  [B, T, C_in, H, W]   (e.g. CIFAR-10: C_in=3, H=W=32)
    Output: [B, T, n_classes]
    """
    def __init__(self, in_channels=3, n_classes=10, T=4):
        super().__init__()
        self.T = T
        self.encoder = TimeDistributed(
            nn.Sequential(
                nn.Conv2d(in_channels, 3, 1, bias=False),   # pixel→rate encoder
                nn.BatchNorm2d(3),
            )
        )
        self.block1 = ConvLIFBlock(3,   64,  3, 1, 1)
        self.pool1  = TimeDistributed(nn.AvgPool2d(2))       # 32→16

        self.block2 = ConvLIFBlock(64,  128, 3, 1, 1)
        self.pool2  = TimeDistributed(nn.AvgPool2d(2))       # 16→8

        self.block3 = ConvLIFBlock(128, 256, 3, 1, 1)
        self.pool3  = TimeDistributed(nn.AvgPool2d(2))       # 8→4

        # 256 * 4 * 4 = 4096
        self.fc = TimeDistributed(nn.Linear(256 * 4 * 4, n_classes))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def named_conv_layers(self):
        """Return (name, Conv2d) pairs for all conv layers."""
        return [(n, m) for n, m in self.named_modules() if isinstance(m, nn.Conv2d)]

    def named_lif_layers(self):
        """Return (name, LIFSpike) pairs for all LIF layers."""
        return [(n, m) for n, m in self.named_modules() if isinstance(m, LIFSpike)]

    def forward(self, x):
        # x: [B, C, H, W]  → repeat over T → [B, T, C, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(1).expand(-1, self.T, -1, -1, -1)

        x = self.encoder(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        B, T, C, H, W = x.shape
        x = x.flatten(2)           # [B, T, C*H*W]
        x = self.fc(x)             # [B, T, n_classes]
        return x                   # mean over T done in training loops
