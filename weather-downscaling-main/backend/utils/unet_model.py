"""
utils/unet_model.py
--------------------
Small residual U-Net matching the architecture described in the project
README ("~150k-parameter residual U-Net ... learns the correction to the
bilinear baseline"). This lets ModelLoader load `models/best_model.pt`
(produced by the original repo's `train.py`) with a matching state_dict.

If your actual checkpoint was saved with a different class definition,
swap this file's `ResidualUNet` for the real one from the training repo --
the rest of the backend (ModelLoader / Predictor) only depends on the
class name and the forward(x) -> tensor contract below.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualUNet(nn.Module):
    """
    Predicts a residual correction to the bilinear baseline; the caller is
    expected to add that residual back onto the baseline channel themselves
    (final = baseline + residual), matching the README's residual-learning
    design so the model can never do worse than the baseline by construction.
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 16):
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock(base_ch * 2, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = _ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)  # residual correction, same H×W as input
