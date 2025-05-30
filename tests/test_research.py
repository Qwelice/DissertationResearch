import torch

from research.modeling.layers.adaconv import AdaptiveConv2d
from research.modeling.layers.attention import L2MultiHeadAttention
from research.modeling.layers.generator import VoxelFormer, GeneratorLayer


def test_generator_layer():
    t = torch.randn(32, 512, 4, 4)
    style = torch.randn(32, 256)
    t_local = torch.randn(32, 64, 256)
    adaconv = AdaptiveConv2d(512, 256, 256, kernel_size=3, stride=1, padding=1)
    s_attn = L2MultiHeadAttention(256, 8)
    x_attn = L2MultiHeadAttention(256, 8)
    vox_former = VoxelFormer(4, 256, 256, 8, 1024, tiq_qk=True)
    generator_layer = GeneratorLayer(input_size=4 * 4,
                                     adaconv=adaconv,
                                     voxel_former=vox_former,
                                     self_atten=s_attn,
                                     cross_atten=x_attn)
    vox, fs = generator_layer(t, style, t_local)
    print(vox.shape, fs.shape)