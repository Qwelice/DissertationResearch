from typing import Optional

import torch
from torch import nn

from research.modeling.layers.adaconv import AdaptiveConv2d
from research.modeling.layers.attention import L2MultiHeadAttention
from research.utils.functions import split_into_patches, merge_patches


class Predictor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, voxel_size: int, style_dim: int):
        super(Predictor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.voxel_size = voxel_size
        self.style_dim = style_dim
        self.conv_1 = AdaptiveConv2d(in_channels, out_channels, style_dim, kernel_size=1, stride=1)
        self.conv_2 = AdaptiveConv2d(out_channels, out_channels, style_dim, kernel_size=1, stride=1)
        self.conv_3 = AdaptiveConv2d(out_channels, out_channels, style_dim, kernel_size=1, stride=1)
        self.conv_4 = AdaptiveConv2d(out_channels, out_channels, style_dim, kernel_size=1, stride=1)
        self.residual = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.fc = nn.Linear(out_channels * voxel_size**2, 1)
        self.sigma = nn.Sigmoid()
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        x = self.leaky(self.conv_1(x, style))
        x = self.leaky(self.conv_2(x, style))
        x = self.leaky(self.conv_3(x, style))
        x = self.leaky(self.conv_4(x, style))
        x = x + self.residual(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.sigma(x)
        return x


class DiscriminatorLayer(nn.Module):
    """
    DO NOT FORGET: [PATCH SIZE YOU'RE USING AFTER DOWNSAMPLING!
                    INPUT SIZE YOU'RE USING BEFORE DOWNSAMPLING]
    """
    def __init__(self, input_size: int, patch_size: int, conv: nn.Conv2d,
                 self_atten: Optional[L2MultiHeadAttention]=None, activation: Optional[str]=None):
        super(DiscriminatorLayer, self).__init__()
        self.input_size = input_size // 2 * patch_size * patch_size
        self.patch_size = patch_size
        self.conv = conv
        self.to_tokens = nn.Linear(self.input_size, self_atten.embed_dim) if self_atten is not None else None
        self.from_tokens = nn.Linear(self_atten.embed_dim, self.input_size) if self_atten is not None else None
        self.self_atten = self_atten
        if activation is None:
            activation = 'relu'
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

    def _self_attn(self, x):
        if self.self_atten is None:
            return x
        x = split_into_patches(x, patch_size=self.patch_size)
        x = self.to_tokens(x)
        x, _ = self.self_atten(x, x, x)
        x = self.from_tokens(x)
        x = merge_patches(x, self.patch_size)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self._self_attn(x)
        return x