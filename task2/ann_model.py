"""
ann_model.py
============
ANN equivalent of SmallSNN — identical architecture (3 conv blocks + FC)
but with ReLU activations instead of LIF spiking neurons.

This is the control/baseline. By keeping architecture identical, any
difference in representation drift between ANN and SNN is attributable
purely to:
  1. The surrogate gradient approximation (SNN-specific)
  2. The sparsity of spiking activity (SNN-specific)

Input:  [B, C, H, W]       (standard image batch, no time dimension)
Output: [B, n_classes]
"""

import torch
import torch.nn as nn


class ConvReLUBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallANN(nn.Module):
    """
    Mirror of SmallSNN with ReLU instead of LIF.
    3 conv blocks, global avg pool, 1 FC.
    """
    def __init__(self, in_channels=3, n_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 3, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.block1 = ConvReLUBlock(3,   64,  3, 1, 1)
        self.pool1  = nn.AvgPool2d(2)    # 32→16

        self.block2 = ConvReLUBlock(64,  128, 3, 1, 1)
        self.pool2  = nn.AvgPool2d(2)    # 16→8

        self.block3 = ConvReLUBlock(128, 256, 3, 1, 1)
        self.pool3  = nn.AvgPool2d(2)    # 8→4

        self.fc = nn.Linear(256 * 4 * 4, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = x.flatten(1)
        return self.fc(x)

    def get_representations(self, x):
        """
        Returns intermediate feature maps at each block for a batch.
        Used by the representation drift tracker.
        Returns dict: layer_name -> flattened feature vector [B, D]
        """
        reps = {}
        x = self.encoder(x)

        x = self.block1(x)
        reps['block1'] = x.flatten(1).detach().cpu()
        x = self.pool1(x)

        x = self.block2(x)
        reps['block2'] = x.flatten(1).detach().cpu()
        x = self.pool2(x)

        x = self.block3(x)
        reps['block3'] = x.flatten(1).detach().cpu()

        return reps

    def get_sparsity(self, x):
        """
        Returns fraction of zero activations per block (sparsity metric).
        ReLU zeros = dead activations. For ANN this is the baseline sparsity.
        """
        sparsity = {}
        x = self.encoder(x)

        x = self.block1(x)
        sparsity['block1'] = (x == 0).float().mean().item()
        x = self.pool1(x)

        x = self.block2(x)
        sparsity['block2'] = (x == 0).float().mean().item()
        x = self.pool2(x)

        x = self.block3(x)
        sparsity['block3'] = (x == 0).float().mean().item()

        return sparsity
