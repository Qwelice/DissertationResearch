import math
from typing import Optional

import torch
from torch import nn


from projects.AVoT.utils import qk_l2_distance


class L2MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 kdim: Optional[int]=None, vdim: Optional[int]=None, tie_qk: bool=True):
        super(L2MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads

        self.tie_qk = tie_qk
        if tie_qk:
            self.kdim = embed_dim

        if embed_dim % self.num_heads != 0:
            raise ValueError('total queries dimension must be divisible by number of attention heads')
        if self.kdim % self.num_heads != 0:
            raise ValueError('total keys dimension must be divisible by number of attention heads')
        if self.vdim % self.num_heads != 0:
            raise ValueError('total values dimension must be divisible by number of attention heads')

        if tie_qk:
            shared = nn.Linear(embed_dim, embed_dim)
            self.q_in_proj = shared
            self.k_in_proj = shared
        else:
            self.q_in_proj = nn.Linear(embed_dim, embed_dim)
            self.k_in_proj = nn.Linear(embed_dim, self.kdim)
        self.v_in_proj = nn.Linear(embed_dim, self.vdim)

        self._scale = 1.0 / math.sqrt(embed_dim)
        self._alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.out_proj = nn.Linear(self.vdim, embed_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        q = self.q_in_proj(queries).unflatten(-1, [self.num_heads, self.embed_dim // self.num_heads]).transpose(1, 2)
        k = self.k_in_proj(keys).unflatten(-1, [self.num_heads, self.kdim // self.num_heads]).transpose(1, 2)
        v = self.v_in_proj(values).unflatten(-1, [self.num_heads, self.vdim // self.num_heads]).transpose(1, 2)
        attn_dist = qk_l2_distance(q, k)
        attn_weights = torch.softmax(self._alpha * torch.exp(-attn_dist) * self._scale, dim=-1)
        attention = attn_weights @ v
        attention = attention.transpose(1, 2).flatten(-2)
        attn_out = self.out_proj(attention)
        return attn_out


class L2TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 activation: Optional[str]=None, tiq_qk: bool=True):
        super(L2TransformerEncoderLayer, self).__init__()
        if activation == 'relu' or activation is None:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.mha = L2MultiHeadAttention(d_model, nhead, tie_qk=tiq_qk)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                self.activation,
                                nn.Linear(dim_feedforward, d_model))

    def forward(self, x):
        x = self.norm_1(x)
        x = x + self.mha(x, x, x)
        x = x + self.fc(self.norm_2(x))
        return x


class L2TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 activation: Optional[str]=None, tie_qk: bool=True):
        super(L2TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        if activation is None or activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
        self.self_attn = L2MultiHeadAttention(embed_dim=d_model,
                                              num_heads=nhead,
                                              tie_qk=tie_qk)
        self.mha = L2MultiHeadAttention(embed_dim=d_model,
                                        num_heads=nhead,
                                        tie_qk=tie_qk)
        self.fc = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                self.activation,
                                nn.Linear(dim_feedforward, d_model))

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

    def _self_attn_block(self, x):
        x = self.self_attn(x, x, x)
        return x

    def _mha_block(self, x, mem):
        x = self.mha(x, mem, mem)
        return x

    def forward(self, tgt, memory):
        tgt = self.norm_1(tgt)
        tgt = tgt + self._self_attn_block(tgt)
        tgt = self.norm_2(tgt)
        tgt = tgt + self._mha_block(tgt, memory)
        tgt = tgt + self.fc(self.norm_3(tgt))
        return tgt