from typing import Optional

import torch
from torch import nn

from research.modeling.layers.adaconv import AdaptiveConv2d
from research.modeling.layers.attention import L2MultiHeadAttention
from research.modeling.layers.transformer import L2TransformerDecoderLayer
from research.modeling.models.transformer import L2TransformerDecoder
from research.utils.enums import AttentionType
from research.utils.functions import get_1d_sin_cos_positional_encoding


class UpsamplingLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, style_dim: int,
                 hidden_channels: Optional[int]=None, bank_size: Optional[int]=4):
        super(UpsamplingLayer, self).__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        self.conv_1 = AdaptiveConv2d(in_channels, hidden_channels, style_dim=style_dim,
                                     kernel_size=3, stride=1, padding=1, bank_size=bank_size)
        self.conv_2 = AdaptiveConv2d(hidden_channels, out_channels, style_dim=style_dim,
                                     kernel_size=3, stride=1, padding=1, bank_size=bank_size)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.activation = nn.ReLU()

    def forward(self, x, style):
        x = self.conv_1(x, style)
        x = self.upsample(x)
        x = self.conv_2(x, style)
        x = self.activation(x)
        return x


class VoxelFormer(nn.Module):
    def __init__(self, voxel_size: int, seq_size: int, emb_dim: int, nhead: int, num_layers: int, dropout: float=0.,
                 attn_type: AttentionType=AttentionType.none, activation: Optional[str]=None, tiq_qk: Optional[bool]=None):
        super(VoxelFormer, self).__init__()
        if attn_type == AttentionType.none:
            raise ValueError('attention type cannot be none')

        self.queries = nn.Parameter(torch.randn(1, seq_size, emb_dim), requires_grad=True)
        if activation is None:
            activation = 'relu'

        if attn_type == AttentionType.attention:
            decoding_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=4 * emb_dim,
                                                        dropout=dropout, batch_first=True, norm_first=True, activation=activation)
            self.decoder = nn.TransformerDecoder(decoding_layer, num_layers=num_layers)
        else:
            decoding_layer = L2TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=4 * emb_dim,
                                                       dropout=dropout, activation=activation, tie_qk=tiq_qk)
            self.decoder = L2TransformerDecoder(decoding_layer, num_layers=num_layers)

        self.pos = get_1d_sin_cos_positional_encoding(seq_size, emb_dim)

        self.x = nn.Linear(emb_dim, voxel_size)
        self.y = nn.Linear(emb_dim, voxel_size)
        self.z = nn.Linear(emb_dim, voxel_size)

    def _one_rank_product(self, u, v, w) -> torch.Tensor:
        P = torch.einsum('bki,bkj,bkl->bijl', u, v, w)
        P = torch.minimum(torch.tensor(1.0), P)
        return P

    def forward(self, t):
        b, _, _ = t.size()
        queries = self.queries.expand(b, -1, -1)
        queries = queries + self.pos.to(t.device)
        t = self.decoder(t, queries)
        x = self.x(t)
        y = self.y(t)
        z = self.z(t)
        voxel = self._one_rank_product(x, y, z)
        return voxel


class GeneratorLayer(nn.Module):
    def __init__(self, voxel_size: int, in_channels: int, out_channels: int, hidden_channels: int,
                 nhead: int, emb_dim: int, style_dim: int, decoding_layers: int, bank_size: int=4,
                 dropout: float=0., attn_type: AttentionType=AttentionType.none,
                 need_patching: bool=False, patch_size: Optional[int]=None):
        super(GeneratorLayer, self).__init__()
        self.need_patching = need_patching
        if need_patching:
            if patch_size is None:
                raise ValueError('patch size must be int if needing patching')
            self.threshold_factor = patch_size
            self.patchify = nn.Conv2d(hidden_channels, emb_dim,
                                      kernel_size=self.threshold_factor, stride=self.threshold_factor)
            self.unpatchify = nn.ConvTranspose2d(emb_dim, hidden_channels,
                                                 kernel_size=self.threshold_factor, stride=self.threshold_factor)

        self.upsampler = UpsamplingLayer(in_channels, hidden_channels, style_dim=style_dim,
                                         hidden_channels=hidden_channels, bank_size=bank_size)
        self.out_conv = AdaptiveConv2d(hidden_channels, out_channels, style_dim=style_dim,
                                       kernel_size=3, stride=1, padding=1, bank_size=bank_size)
        self.attn_type = attn_type
        voxel_former_attn_type = attn_type if attn_type != AttentionType.none else AttentionType.attention
        self.voxel_former = VoxelFormer(voxel_size, voxel_size, emb_dim=emb_dim, nhead=nhead,
                                        num_layers=decoding_layers, dropout=dropout, attn_type=voxel_former_attn_type)
        if attn_type != AttentionType.none:
            if attn_type == AttentionType.attention:
                self.self_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=nhead, dropout=dropout,
                                                       batch_first=True)
                self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=nhead, dropout=dropout,
                                                        batch_first=True)
            else:
                self.self_attn = L2MultiHeadAttention(embed_dim=emb_dim, num_heads=nhead, dropout=dropout, tie_qk=True)
                self.cross_attn = L2MultiHeadAttention(embed_dim=emb_dim, num_heads=nhead, dropout=dropout,
                                                       tie_qk=True)

            self.norm_1 = nn.LayerNorm(emb_dim)
            self.norm_2 = nn.LayerNorm(emb_dim)

        self.norm_3 = nn.LayerNorm(emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )

    def self_attention(self, x):
        if self.attn_type != AttentionType.none:
            x2 = self.norm_1(x)
            x2, _ = self.self_attn(x2, x2, x2)
            x = x + x2
        return x

    def cross_attention(self, x, t_local):
        if self.attn_type != AttentionType.none:
            x2 = self.norm_2(x)
            x2, _ = self.cross_attn(x2, t_local, t_local)
            x = x2 + x
        return x

    def ffn(self, x):
        x2 = self.norm_3(x)
        x2 = self.fc(x2)
        x = x + x2
        return x

    def forward(self, x, style, t_local):
        x = self.upsampler(x, style)
        if self.need_patching:
            x = self.patchify(x)
        B, C, H, W = x.shape
        L = H * W
        flatten = x.view(B, C, L).permute(0, 2, 1) # B, L, C

        x = self.self_attention(flatten)
        x = self.cross_attention(x, t_local)
        x = self.ffn(x)

        features = x.permute(0, 2, 1).view(B, C, H, W)
        if self.need_patching:
            features = self.unpatchify(features)
        features = self.out_conv(features, style)
        out = self.voxel_former(x)

        return out, features