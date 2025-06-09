import copy

from torch import nn

from research.modeling.layers.transformer import L2TransformerEncoderLayer, L2TransformerDecoderLayer


class L2TransformerEncoder(nn.Module):
    def __init__(self, enc_layer: L2TransformerEncoderLayer, num_layers: int):
        super(L2TransformerEncoder, self).__init__()
        self.layers = self._get_clones(enc_layer, num_layers)
        self.enc_layer = enc_layer
        self.num_layers = num_layers

    def _get_clones(self, module: nn.Module, num: int):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])

    def forward(self, x):
        out = x
        for mod in self.layers:
            out = mod(out)
        return out


class L2TransformerDecoder(nn.Module):
    def __init__(self, dec_layer: L2TransformerDecoderLayer, num_layers: int):
        super(L2TransformerDecoder, self).__init__()
        self.layers = self._get_clones(dec_layer, num_layers)
        self.dec_layer = dec_layer
        self.num_layers = num_layers

    def _get_clones(self, module: nn.Module, num: int):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])

    def forward(self, tgt, memory):
        out = tgt
        for mod in self.layers:
            out = mod(out, memory)
        return out