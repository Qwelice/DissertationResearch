from typing import Dict

from torch import nn

from research.modeling.models.image_encoder import ImageEncoder
from research.modeling.models.mapping_net import MappingNet
from research.utils.enums import LayerType


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.layers = self._init_layers_()
        self.image_encoder = ImageEncoder(config)
        self.mapping_net = MappingNet(config)

    def _init_layers_(self) -> nn.ModuleList:
        cfg = self.config.generator
        layers = []
        for layer in cfg.layers:
            tp: LayerType = layer['type']
            params: Dict = {
                **layer['params'],
                **self._get_layer_params(layer)
            }
            module =  tp(**params)
            if module is None:
                raise ValueError('module cannot be None')
            else:
                layers.append(module)
        return nn.ModuleList(layers)

    def _get_layer_params(self, layer_config: Dict) -> nn.Module:
        parameters = {}
        layers = layer_config['layers']
        for layer in layers:
            tp = layer['type']
            params = layer['params']
            module = tp(**params)
            if tp == LayerType.AdaConv2d:
                parameters['adaconv'] = module
            elif tp == LayerType.VoxelFormer:
                parameters['voxel_former'] = module
            elif tp == LayerType.SelfL2Attention or LayerType.SelfAttention:
                parameters['self_atten'] = module
            elif tp == LayerType.CrossL2Attention or LayerType.CrossAttention:
                parameters['cross_atten'] = module
        return parameters

    def forward(self, x, style, t_local):
        outs = []
        for layer in self.layers:
            out, x = layer(x, style, t_local)
            outs.append(out)
        return outs