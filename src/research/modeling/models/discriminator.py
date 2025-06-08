from typing import Dict

from torch import nn

from research.modeling.models.image_encoder import ImageEncoder
from research.utils.constants import LayerInitMap
from research.utils.enums import LayerType


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.layers = self._init_layers_()
        self.predictors = self._init_predictors()
        assert len(self.layers) == len(self.predictors), 'discriminator layers count must match to predictors count'
        self.image_encoder = ImageEncoder(config)

    def _init_predictors(self) -> nn.ModuleList:
        cfg = self.config.discriminator
        predictors = []
        for layer in cfg.predictors:
            tp = layer['type']
            init_fn = LayerInitMap[tp]
            params = layer['params']
            module = init_fn(**params)
            if module is None:
                raise ValueError('module cannot be None')
            else:
                predictors.append(module)
        return nn.ModuleList(predictors)

    def _init_layers_(self) -> nn.ModuleList:
        cfg = self.config.discriminator
        layers = []
        for layer in cfg.layers:
            tp: LayerType = layer['type']
            init_fn = LayerInitMap[tp]
            params: Dict = {
                **layer['params'],
                **self._get_layer_params(layer)
            }
            module = init_fn(**params)
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
            init_fn = LayerInitMap[tp]
            module = init_fn(**params)
            if tp == LayerType.Predictor:
                key = 'predictor'
            elif tp == LayerType.Conv2d:
                key = 'conv'
            elif tp == LayerType.SelfL2Attention:
                key = 'self_atten'
            else:
                raise ValueError(f'unknown parameter: {tp}')
            parameters[key] = module
        return parameters

    def get_descriptor(self, x):
        descriptor = self.image_encoder(x)
        return descriptor

    def forward(self, x, t_global):
        """ Feed forward method

        Args:
            x: the sequence of different scaled voxels
            t_global: local descriptor extracted from image through image encoder

        """
        outs = []
        N = len(self.layers)
        for i in range(N):
            preds = []
            phi = x[i]
            for j in range(i, N):
                phi = self.layers[j](phi)
                psi = self.predictors[j](phi, t_global)
                preds.append(psi)
            outs.append(preds)
        return outs