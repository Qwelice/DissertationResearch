from typing import Dict

import torch
from torch import nn

from research.modeling.models.image_encoder import ImageEncoder
from research.modeling.models.mapping_net import MappingNet
from research.utils.constants import LayerInitMap, WeightsInitMap, ParametersInitMap
from research.utils.enums import LayerType


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.layers = self._init_layers_()
        self.image_encoder = ImageEncoder(config)
        self.mapping_net = MappingNet(config)
        self.base_features = self._init_base_features_()

    def _init_base_features_(self) -> nn.Parameter:
        cfg = self.config.generator
        init_fn = ParametersInitMap[cfg.base_features['weights_init']]
        init_params = cfg.base_features['weights_init_params']
        base_shape = cfg.base_features['shape']
        base_features = nn.Parameter(torch.zeros(1, *base_shape))
        init_fn(base_features, **init_params)
        return base_features

    def _init_layers_(self) -> nn.ModuleList:
        cfg = self.config.generator
        layers = []
        for layer in cfg.layers:
            tp: LayerType = layer['type']
            init_fn = LayerInitMap[tp]
            params = layer['params']
            module =  init_fn(**params)
            if module is None:
                raise ValueError('module cannot be None')
            else:
                layers.append(module)
        return nn.ModuleList(layers)

    def get_style(self, t_global):
        style = self.mapping_net(t_global)
        return style

    def get_descriptor(self, x):
        descriptor = self.image_encoder(x)
        return descriptor

    def forward(self, style, t_local):
        bs, _ = style.shape
        device = style.device
        x = self.base_features.expand(bs, -1, -1, -1).contiguous().to(device)
        outs = []
        for layer in self.layers:
            out, x = layer(x, style, t_local)
            outs.append(out)
        return outs