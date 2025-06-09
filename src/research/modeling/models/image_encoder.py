from torch import nn

from research.utils.constants import LayerInitMap
from research.utils.enums import LayerType


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.config = config
        self.layers = self._init_layers_()

    def _init_layers_(self):
        cfg = self.config.image_encoder
        image_encoder = []
        for layer in cfg.layers:
            tp: LayerType = layer['type']
            params = layer['params']
            init_fn = LayerInitMap[tp]
            if tp == LayerType.L2Encoder:
                sub_layer = layer['layers'][0]
                enc_tp = sub_layer['type']
                enc_params = sub_layer['params']
                sub_init_fn = LayerInitMap[enc_tp]
                sub_module = sub_init_fn(**enc_params)
                params = {
                    **params,
                    'enc_layer': sub_module
                }
            module = init_fn(**params)
            if module is None:
                raise ValueError('module cannot be None')
            else:
                image_encoder.append(module)
        return nn.Sequential(*image_encoder)

    def forward(self, x):
        x = self.layers(x)
        return x