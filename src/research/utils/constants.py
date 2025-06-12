from research.utils.enums import WeightsInitType, OptimizerType, LayerType
from research.utils.initializers.layer_initializers import *
from research.utils.initializers.optimizer_initializers import *
from research.utils.initializers.weights_initializers import *

LayerInitMap = {
    LayerType.Linear: linear,
    LayerType.ReLU: relu,
    LayerType.Dropout: dropout,
    LayerType.AdapterLayer: adapter_layer,
    LayerType.Conv2d: conv2d,
    LayerType.L2Attention: l2attention,
    LayerType.L2EncoderLayer: l2encoder_layer,
    LayerType.L2DecoderLayer: l2decoder_layer,
    LayerType.L2Encoder: l2encoder,
    LayerType.L2Decoder: l2decoder,
    LayerType.AdaConv2d: adaconv2d,
    LayerType.GELU: gelu,
    LayerType.Leaky: leaky,
    LayerType.GeneratorLayer: generator_layer,
    LayerType.DiscriminatorLayer: discriminator_layer,
    LayerType.Predictor: predictor,
    LayerType.SelfL2Attention: l2attention,
    LayerType.CrossL2Attention: l2attention,
    LayerType.SelfAttention: attention,
    LayerType.CrossAttention: attention,
    LayerType.ResNet18: resnet18
}

WeightsInitMap = {
    WeightsInitType.normal: init_weights_normal,
    WeightsInitType.uniform: init_weights_uniform,
    WeightsInitType.xavier_uniform: init_weights_xavier_uniform,
    WeightsInitType.xavier_normal: init_weights_xavier_normal,
    WeightsInitType.kaiming_uniform: init_weights_kaiming_uniform,
    WeightsInitType.kaiming_normal: init_weights_kaiming_normal
}

ParametersInitMap = {
    WeightsInitType.normal: init.normal_,
    WeightsInitType.uniform: init.uniform_,
    WeightsInitType.xavier_uniform: init.xavier_uniform_,
    WeightsInitType.xavier_normal: init.xavier_normal_,
    WeightsInitType.kaiming_uniform: init.kaiming_uniform_,
    WeightsInitType.kaiming_normal: init.kaiming_normal_
}

OptimizersInitMap = {
    OptimizerType.adam: adam
}