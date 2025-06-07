from abc import ABC
from enum import IntEnum, Enum

from research.utils.layer_initializers import *


class CallableEnum(ABC, Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

class LayerType(CallableEnum):
    Linear = linear,
    ReLU = relu,
    Dropout = dropout,
    AdapterLayer = adapter_layer,
    Conv2d = conv2d,
    L2Attention = l2attention,
    L2Encoder = l2encoder,
    L2Decoder = l2decoder,
    AdaConv2d = adaconv2d,
    GELU = gelu,
    Leaky = leaky,
    GeneratorLayer = generator_layer,
    DiscriminatorLayer = discriminator_layer,
    VoxelFormer = voxel_former,
    Predictor = predictor,
    VoxelAdapter = voxel_adapter,
    SelfL2Attention = l2attention,
    CrossL2Attention = l2attention,
    SelfAttention = attention,
    CrossAttention = attention,
    ResNet18 = resnet18


ConversionType = IntEnum('ConversionType', 'split', 'merge')

SetType = IntEnum('SetType', ('train', 'eval', 'test'))
ReductionType = IntEnum('ReductionType', ('max', 'avg'))
WeightsInitType = IntEnum('WeightsInitType', ('normal', 'uniform', 'xavier_uniform', 'xavier_normal',
                                              'kaiming_uniform', 'kaiming_normal'))
OptimizerType = IntEnum('OptimizerType', ('sgd', 'adam'))
SchedulerType = IntEnum('SchedulerType', ('cosine_annealing', 'none'))


def recognize_set_type(value: str) -> SetType:
    for tp in SetType:
        if tp.name == value:
            return tp
    raise ValueError(f"set type `{value}` didn't recognize")