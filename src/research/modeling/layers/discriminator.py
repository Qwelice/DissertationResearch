from typing import Optional

import torch
from torch import nn

from research.modeling.layers.adaconv import AdaptiveConv2d
from research.modeling.layers.attention import L2MultiHeadAttention
from research.utils.enums import AttentionType


class Predictor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, voxel_size: int, style_dim: int):
        super(Predictor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.voxel_size = voxel_size
        self.style_dim = style_dim
        self.conv_1 = AdaptiveConv2d(in_channels, out_channels, style_dim, kernel_size=1, stride=1, bank_size=4)
        self.conv_2 = AdaptiveConv2d(out_channels, out_channels, style_dim, kernel_size=1, stride=1, bank_size=4)
        # self.conv_3 = AdaptiveConv2d(out_channels, out_channels, style_dim, kernel_size=1, stride=1, bank_size=4)
        # self.conv_4 = AdaptiveConv2d(out_channels, out_channels, style_dim, kernel_size=1, stride=1, bank_size=4)
        self.unconditional = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.fc = nn.Linear(out_channels * voxel_size**2, 1)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        conditional = self.leaky(self.conv_1(x, style))
        conditional = self.leaky(self.conv_2(conditional, style))
        # conditional = self.leaky(self.conv_3(conditional, style))
        # conditional = self.leaky(self.conv_4(conditional, style))
        unconditional = self.leaky(self.unconditional(x))
        x = conditional + unconditional
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class DiscriminatorLayer(nn.Module):
    def __init__(self, voxel_size: int, out_channels: int, emb_dim: int,
                 dropout: float=0., nhead: Optional[int]=None, attn_type: AttentionType=AttentionType.none):
        super(DiscriminatorLayer, self).__init__()
        assert voxel_size % 2 == 0, 'voxel size must be divisible by 2'

        self.conv = nn.Conv2d(voxel_size, emb_dim, kernel_size=3, stride=2, padding=1)
        self.features_conv = nn.Conv2d(emb_dim, out_channels, kernel_size=3, stride=1, padding=1)
        self.voxel_conv = nn.Conv2d(emb_dim, voxel_size // 2, kernel_size=3, stride=1, padding=1)

        self.attn_type = attn_type
        if attn_type != AttentionType.none:
            if attn_type == AttentionType.attention:
                self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=nhead, dropout=dropout, batch_first=True)
            elif attn_type == AttentionType.l2attention:
                self.attn = L2MultiHeadAttention(embed_dim=emb_dim, num_heads=nhead, dropout=dropout, tie_qk=True)
            self.norm_1 = nn.LayerNorm(emb_dim)

        self.norm_2 = nn.LayerNorm(emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )

    def self_attn(self, x):
        if self.attn_type != AttentionType.none:
            x2 = self.norm_1(x)
            x2, _ = self.attn(x2, x2, x2)
            x = x + x2
        return x

    def ffn(self, x):
        x2 = self.norm_2(x)
        x2 = self.fc(x2)
        x = x + x2
        return x

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(1)
        x = self.conv(x)
        B, C, H, W = x.shape
        L = H * W
        flatten = x.view(B, C, L).permute(0, 2, 1)
        x = self.self_attn(flatten)
        x = self.ffn(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        voxel = self.voxel_conv(x)
        features = self.features_conv(x)
        return features, voxel