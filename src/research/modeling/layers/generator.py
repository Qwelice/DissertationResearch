from typing import Optional

import torch
from torch import nn

from research.modeling.layers.adaconv import AdaptiveConv2d
from research.modeling.layers.attention import L2MultiHeadAttention
from research.modeling.layers.transformer import L2TransformerDecoderLayer
from research.utils.functions import split_into_patches, get_2d_sincos_pos_embed, merge_patches


class VoxelFormer(nn.Module):
    def __init__(self, input_size: int, seq_size: int, dim_size: int, nhead: int, dim_feedforward: int,
                 activation: Optional[str]=None, tiq_qk: Optional[bool]=None):
        super(VoxelFormer, self).__init__()
        self.input_size = input_size
        self.seq_size = seq_size
        self.dim_size = dim_size
        self.queries = nn.Parameter(torch.randn(1, seq_size, dim_size), requires_grad=True)
        self.decoder = L2TransformerDecoderLayer(d_model=dim_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                 activation=activation, tie_qk=tiq_qk)
        self.x = nn.Linear(dim_size, input_size)
        self.y = nn.Linear(dim_size, input_size)
        self.z = nn.Linear(dim_size, input_size)

    def _one_rank_product(self, u, v, w) -> torch.Tensor:
        P = torch.einsum('bki,bkj,bkm->bijm', u, v, w)
        P = torch.minimum(torch.tensor(1.0), P)
        return P

    def forward(self, t):
        b, _, _ = t.size()
        queries = self.queries.expand(b, -1, -1)
        t = self.decoder(t, queries)
        x = self.x(t)
        y = self.y(t)
        z = self.z(t)
        voxel = self._one_rank_product(x, y, z)
        return voxel


class GeneratorLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 patch_size: int,
                 adaconv: AdaptiveConv2d,
                 voxel_former: VoxelFormer,
                 self_atten: Optional[L2MultiHeadAttention]=None,
                 cross_atten: Optional[L2MultiHeadAttention]=None,
                 upsample_input: bool=False,
                 size_threshold: int=32):
        super(GeneratorLayer, self).__init__()
        self.input_size = (1 + upsample_input) * input_size
        self.patch_size = patch_size
        self.to_tokens = nn.Linear(self.input_size, self_atten.embed_dim)
        self.from_tokens = nn.Linear(self_atten.embed_dim, self.input_size)
        self.adaconv = adaconv
        self.voxel_former = voxel_former
        self.self_atten = self_atten
        self.cross_atten = cross_atten
        self.upsample = upsample_input
        self.threshold = size_threshold

    def _self_attention(self, x):
        if self.self_atten is None:
            return x
        return self.self_atten(x, x, x)

    def _cross_attention(self, x, t_local):
        if self.cross_atten is None:
            return x
        return self.cross_atten(x, t_local, t_local)

    def forward(self, x, style, t_local):
        if self.upsample:
            x = nn.functional.upsample(x, scale_factor=2, mode='bicubic')
        B, _, H, W = x.size()
        x = self.adaconv(x, style)
        x = split_into_patches(x, self.patch_size)
        pos = get_2d_sincos_pos_embed(x.size(1), x.size(2), x.size(3))
        x = self.to_tokens(x)
        x = x + pos.transpose(1, 0)
        if style.ndim == 2:
            style = style.unsqueeze(1)
        styled = torch.cat([x, style], dim=1)
        x, _ = self._self_attention(styled)
        x = x[::, :-1, ::] # drop style
        x, _ = self._cross_attention(x, t_local)
        voxel = self.voxel_former(x)
        x = self.from_tokens(x)
        features = merge_patches(x, self.patch_size)
        return voxel, features