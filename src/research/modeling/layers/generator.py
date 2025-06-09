from typing import Optional

import torch
from torch import nn

from research.modeling.layers.adaconv import AdaptiveConv2d
from research.modeling.layers.attention import L2MultiHeadAttention
from research.modeling.layers.transformer import L2TransformerDecoderLayer
from research.utils.functions import split_into_patches, get_2d_sin_cos_pos_embed, merge_patches


class VoxelFormer(nn.Module):
    def __init__(self, input_size: int, seq_size: int, dim_size: int, nhead: int, dim_feedforward: int,
                 activation: Optional[str]=None, tiq_qk: Optional[bool]=None, is_l2: Optional[bool]=None):
        super(VoxelFormer, self).__init__()
        self.input_size = input_size
        self.seq_size = seq_size
        self.dim_size = dim_size
        self.queries = nn.Parameter(torch.randn(1, seq_size, dim_size), requires_grad=True)
        is_l2 = is_l2 if is_l2 is not None else False
        if is_l2:
            self.decoder = L2TransformerDecoderLayer(d_model=dim_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                 activation=activation, tie_qk=tiq_qk)
        else:
            self.decoder_layer = nn.TransformerDecoderLayer(d_model=dim_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                      activation='relu', batch_first=True)
            self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
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
    """
    DO NOT FORGET: INPUT SIZE YOU'RE USING FOR FEATURES ON BEFORE UPSAMPLING!
    """
    def __init__(self,
                 input_size: int,
                 patch_size: int,
                 adaconv: AdaptiveConv2d,
                 voxel_former: VoxelFormer,
                 self_atten: Optional[L2MultiHeadAttention]=None,
                 cross_atten: Optional[L2MultiHeadAttention]=None,
                 emb_dim: Optional[int]=None,
                 size_threshold: int=32):
        super(GeneratorLayer, self).__init__()
        self.input_size = 2 * input_size * patch_size * patch_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim if self_atten is None else self_atten.embed_dim

        assert self.emb_dim is not None, ('if self-attention or cross-attention is None'
                                          ' embedding dim must be int, but got None')

        self.to_tokens = nn.Linear(self.input_size, self.emb_dim)
        self.from_tokens = nn.Linear(self.emb_dim, self.input_size)
        self.adaconv = adaconv
        self.voxel_former = voxel_former
        self.self_atten = self_atten
        self.cross_atten = cross_atten
        self.threshold = size_threshold

    def _self_attention(self, x):
        if self.self_atten is None:
            return x
        out, _ = self.self_atten(x, x, x)
        return out

    def _cross_attention(self, x, t_local):
        if self.cross_atten is None:
            return x
        out, _ = self.cross_atten(x, t_local, t_local)
        return out

    def _attention(self, x, style, t_local):
        B, _, H, W = x.shape
        pos = get_2d_sin_cos_pos_embed(H // self.patch_size,
                                       W // self.patch_size,
                                       self.emb_dim).unsqueeze(0).expand(B, -1, -1).to(x.device)
        x = split_into_patches(x, self.patch_size)
        x = self.to_tokens(x)
        x = x + pos
        if style.ndim == 2:
            style = style.unsqueeze(1)
        styled = torch.cat([x, style], dim=1)
        x = self._self_attention(styled)
        x = x[::, :-1, ::]  # drop style
        x = self._cross_attention(x, t_local)
        return x

    def forward(self, x, style, t_local):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.adaconv(x, style)
        x = self._attention(x, style, t_local)
        voxel = self.voxel_former(x)
        x = self.from_tokens(x)
        features = merge_patches(x, self.patch_size)
        return voxel, features