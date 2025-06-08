from torch import nn

from research.modeling.layers.adaconv import AdaptiveConv2d
from research.modeling.layers.adapter import AdapterLayer
from research.modeling.layers.attention import L2MultiHeadAttention
from research.modeling.layers.discriminator import DiscriminatorLayer, Predictor
from research.modeling.layers.generator import GeneratorLayer, VoxelFormer
from research.modeling.layers.transformer import L2TransformerEncoderLayer, L2TransformerDecoderLayer
from research.modeling.models.common import get_resnet
from research.utils.enums import LayerType


def linear(**params) -> nn.Module:
    return nn.Linear(**params)

def resnet18(**params) -> nn.Module:
    resnet_params = {
        'type': LayerType.ResNet18,
        'params': params
    }
    return get_resnet(resnet_params)

def l2attention(**params) -> nn.Module:
    return L2MultiHeadAttention(**params)

def adapter_layer(**params) -> nn.Module:
    return AdapterLayer(**params)

def adaconv2d(**params) -> nn.Module:
    return AdaptiveConv2d(**params)

def conv2d(**params) -> nn.Module:
    return nn.Conv2d(**params)

def relu(**params) -> nn.Module:
    return nn.ReLU(**params)

def gelu(**params) -> nn.Module:
    return nn.GELU(**params)

def attention(**params) -> nn.Module:
    return nn.MultiheadAttention(**params)

def l2encoder(**params) -> nn.Module:
    return L2TransformerEncoderLayer(**params)

def l2decoder(**params) -> nn.Module:
    return L2TransformerDecoderLayer(**params)

def transformer_encoder(**params) -> nn.Module:
    return nn.TransformerEncoder(**params)

def transformer_decoder(**params) -> nn.Module:
    return nn.TransformerDecoder(**params)

def generator_layer(**params) -> nn.Module:
    return GeneratorLayer(**params)

def discriminator_layer(**params) -> nn.Module:
    return DiscriminatorLayer(**params)

def voxel_former(**params) -> nn.Module:
    return VoxelFormer(**params)

def predictor(**params) -> nn.Module:
    return Predictor(**params)

def dropout(**params) -> nn.Module:
    return nn.Dropout(**params)

def leaky(**params) -> nn.Module:
    return nn.LeakyReLU(**params)