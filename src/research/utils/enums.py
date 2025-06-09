from enum import IntEnum


LayerType = IntEnum('LayerType', (
    'Linear', 'ReLU', 'Dropout', 'AdapterLayer', 'Conv2d', 'L2Attention', 'L2EncoderLayer', 'L2DecoderLayer',
    'AdaConv2d', 'GELU', 'Leaky', 'GeneratorLayer', 'DiscriminatorLayer', 'VoxelFormer', 'Predictor',
    'SelfL2Attention', 'CrossL2Attention', 'SelfAttention', 'CrossAttention', 'ResNet18', 'L2Encoder', 'L2Decoder'
))
ConversionType = IntEnum('ConversionType', ('split', 'merge'))

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