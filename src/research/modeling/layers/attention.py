import math
from typing import Optional

import torch
from torch import nn


def _qk_l2_distance(queries: torch.Tensor, keys: torch.Tensor):
    q = queries.pow(2).sum(dim=-1, keepdim=True)
    k = keys.pow(2).sum(dim=-1, keepdim=True)
    dist = q + k.transpose(-2, -1) - 2 * torch.matmul(queries, keys.transpose(-2, -1))
    return dist


class L2MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.,
                 kdim: Optional[int]=None, vdim: Optional[int]=None, tie_qk: bool=True):
        super(L2MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

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
        self.out_proj = nn.Linear(self.vdim, embed_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor]=None):
        q = self.q_in_proj(queries).unflatten(-1, [self.num_heads, self.embed_dim // self.num_heads]).transpose(1, 2)
        k = self.k_in_proj(keys).unflatten(-1, [self.num_heads, self.kdim // self.num_heads]).transpose(1, 2)
        v = self.v_in_proj(values).unflatten(-1, [self.num_heads, self.vdim // self.num_heads]).transpose(1, 2)
        attn_dist = _qk_l2_distance(q, k)

        if attn_mask is not None:
            attn_dist = attn_dist.masked_fill(attn_mask, float('inf'))

        attn_weights = torch.softmax(-attn_dist * self._scale, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attention = attn_weights @ v
        attention = attention.transpose(1, 2).flatten(-2)
        attn_out = self.out_proj(attention)
        return attn_out, attn_weights