from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'eval', 'test'))
LayerType = IntEnum('LayerType',
                    ('Linear', 'ReLU', 'Dropout', 'Conv2d', 'L2Attention', 'L2Encoder', 'L2Decoder', 'AdaConv2d','GeLU'))
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