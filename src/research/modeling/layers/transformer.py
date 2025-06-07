from typing import Optional

from torch import nn

from research.modeling.layers.attention import L2MultiHeadAttention


class L2TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 activation: Optional[str]=None, tiq_qk: bool=True):
        super(L2TransformerEncoderLayer, self).__init__()
        if activation is None or activation.lower() == 'relu':
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

    def _self_attn(self, x):
        x, _ = self.mha(x, x, x)
        return x

    def forward(self, x):
        x = self.norm_1(x)
        x = x + self._self_attn(x)
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
        x, _ = self.self_attn(x, x, x) # no need for attention weights
        return x

    def _mha_block(self, x, mem):
        x, _ = self.mha(x, mem, mem) # no need for attention weights
        return x

    def forward(self, tgt, memory):
        tgt = self.norm_1(tgt)
        tgt = tgt + self._self_attn_block(tgt)
        tgt = self.norm_2(tgt)
        tgt = tgt + self._mha_block(tgt, memory)
        tgt = tgt + self.fc(self.norm_3(tgt))
        return tgt