import torch
from torch import nn

from research.utils.constants import LayerInitMap
from research.utils.enums import LayerType


class MappingNet(nn.Module):
    def __init__(self, config):
        super(MappingNet, self).__init__()
        self.config = config
        self.layers = self._init_layers_()
        self.latent_dim = config.latent_dim

    def _init_layers_(self):
        cfg = self.config.image_encoder
        image_encoder = []
        for layer in cfg.layers:
            tp: LayerType = layer['type']
            params = layer['params']
            init_fn = LayerInitMap[tp]
            module = init_fn(params)
            if module is None:
                raise ValueError('module cannot be None')
            else:
                image_encoder.append(module)
        return nn.Sequential(*image_encoder)

    def forward(self, x):
        device = x.device
        if x.ndim == 3:
            x = x.squeeze(1)
        bs, _ = x.shape
        latent = torch.randn(bs, self.latent_dim, device=device)
        x = torch.cat([x, latent], dim=1)
        x = self.layers(x)
        return x