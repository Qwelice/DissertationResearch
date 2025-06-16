from easydict import EasyDict

from research.utils.enums import LayerType, ConversionType, WeightsInitType, AttentionType
from research.utils.initializers.layer_initializers import dropout

# L2Attention params: <'embed_dim': int,
#                      'num_heads': int,
#                      'kdim': Optional[int]=None,
#                      'vdim': Optional[int]=None,
#                      'tie_qk': bool=True>

# AdaptiveConv2d params: <'in_channels': int,
#                         'out_channels': int,
#                         'style_dim': int,
#                         'kernel_size': int=3,
#                         'stride': int=1,
#                         'padding': int=0,
#                         'bank_size': int=4,
#                         'eps': float=1e-8>

# L2TransformerEncoderLayer params: <'d_model': int,
#                                    'nhead': int,
#                                    'dim_feedforward': int,
#                                    'activation': Optional[str]=None,
#                                    'tiq_qk': bool=True>

# L2TransformerDecoderLayer params: <'d_model': int,
#                                    'nhead': int,
#                                    'dim_feedforward': int,
#                                    'activation': Optional[str]=None,
#                                    'tie_qk': bool=True>

# GeneratorLayer params: <'input_size': int,
#                         'patch_size': int,
#                         'adaconv': AdaptiveConv2d,
#                         'voxel_former': VoxelFormer,
#                         'self_atten': Optional[L2MultiHeadAttention]=None,
#                         'cross_atten': Optional[L2MultiHeadAttention]=None,
#                         'upsample_input': bool=False,
#                         'size_threshold': int=32>

# VoxelFormerLayer params: <'input_size': int,
#                           'seq_size': int,
#                           'dim_size': int,
#                           'nhead': int,
#                           'dim_feedforward': int,
#                           'activation': Optional[str]=None,
#                           'tiq_qk': Optional[bool]=None>

# DiscriminatorLayer params: <'predictor': Predictor,
#                             'voxel_adapter': VoxelAdapter,
#                             'downsample': bool=False>

# VoxelAdapter params: <'in_channels': int,
#                       'out_channels': int,
#                       'patch_size': int,
#                       'emb_dim': int,
#                       'num_heads': int,
#                       'tie_qk': bool=True>

# Predictor params: <'in_channels': int,
#                    'out_channels': int,
#                    'voxel_size': int,
#                    'style_dim': int>

STYLE_DIM = 256
DESCRIPTOR_DIM = 256
LATENT_DIM = 128

model_cfg = EasyDict()

model_cfg.style_dim = STYLE_DIM
model_cfg.descriptor_dim = DESCRIPTOR_DIM
model_cfg.latent_dim = LATENT_DIM

model_cfg.image_encoder = EasyDict()
model_cfg.image_encoder.layers = [
    {
        'type': LayerType.ResNet18,
        'params': {
            'requires_grad': False,
            'drop_last': 2
        },
        'validate': {
            'input': (3, 224, 224),
            'output': (512, 7, 7)
        }
    },
    {
        'type': LayerType.Conv2d,
        'params': {
            'in_channels': 512,
            'out_channels': DESCRIPTOR_DIM,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        },
        'validate': {
            'input': (512, 7, 7),
            'output': (DESCRIPTOR_DIM, 7, 7)
        }
    },
    {
        'type': LayerType.AdapterLayer,
        'params': {
            'conversion_type': ConversionType.split,
            'patch_size': 1
        },
        'validate': {
            'input': (DESCRIPTOR_DIM, 7, 7),
            'output': (49, DESCRIPTOR_DIM)
        }
    },
    {
        'type': LayerType.L2Encoder,
        'params': {
            'num_layers': 4
        },
        'layers': [
            {
                'type': LayerType.L2EncoderLayer,
                'params': {
                    'd_model': DESCRIPTOR_DIM,
                    'nhead': 8,
                    'dim_feedforward': 2048,
                    'activation': 'gelu',
                    'tiq_qk': True
                }
            }
        ],
        'validate': {
            'input': (49, DESCRIPTOR_DIM),
            'output': (49, DESCRIPTOR_DIM)
        }
    }
]

model_cfg.mapping_net = EasyDict()
model_cfg.mapping_net.layers = [
    {
        'type': LayerType.Linear,
        'params': {
            'in_features': DESCRIPTOR_DIM + LATENT_DIM,
            'out_features': 128
        }
    },
    {
        'type': LayerType.ReLU,
        'params': {}
    },
    {
        'type': LayerType.Linear,
        'params': {
            'in_features': 128,
            'out_features': STYLE_DIM
        }
    }
]


model_cfg.generator = EasyDict()
model_cfg.generator.base_features = {
    'shape': (512, 2, 2)
}
model_cfg.generator.layers = [
    {
        'type': LayerType.GeneratorLayer,
        'params': {
            'voxel_size': 4,
            'in_channels': 512,
            'hidden_channels': DESCRIPTOR_DIM,
            'out_channels': 256,
            'nhead': 8,
            'emb_dim': DESCRIPTOR_DIM,
            'style_dim': STYLE_DIM,
            'decoding_layers': 4,
            'attn_type': AttentionType.none,
            'dropout': 0.2,
            'bank_size': 6
        }
    },
    {
        'type': LayerType.GeneratorLayer,
        'params': {
            'voxel_size': 8,
            'in_channels': 256,
            'hidden_channels': DESCRIPTOR_DIM,
            'out_channels': 128,
            'nhead': 8,
            'emb_dim': DESCRIPTOR_DIM,
            'style_dim': STYLE_DIM,
            'decoding_layers': 4,
            'attn_type': AttentionType.attention,
            'dropout': 0.2,
            'bank_size': 6
        }
    },
    {
        'type': LayerType.GeneratorLayer,
        'params': {
            'voxel_size': 16,
            'in_channels': 128,
            'hidden_channels': DESCRIPTOR_DIM,
            'out_channels': 64,
            'nhead': 8,
            'emb_dim': DESCRIPTOR_DIM,
            'style_dim': STYLE_DIM,
            'decoding_layers': 4,
            'attn_type': AttentionType.none,
            'dropout': 0.2,
            'bank_size': 6
        }
    }
]

model_cfg.discriminator = EasyDict()
model_cfg.discriminator.layers = [
    {
        'type': LayerType.DiscriminatorLayer,
        'params': {
            'voxel_size': 16,
            'out_channels': 32,
            'emb_dim': DESCRIPTOR_DIM,
            'dropout': 0.5,
            'attn_type': AttentionType.l2attention,
            'nhead': 4
        }
    },
    {
        'type': LayerType.DiscriminatorLayer,
        'params': {
            'voxel_size': 8,
            'out_channels': 64,
            'emb_dim': DESCRIPTOR_DIM,
            'dropout': 0.5,
            'attn_type': AttentionType.l2attention,
            'nhead': 4
        }
    },
    {
        'type': LayerType.DiscriminatorLayer,
        'params': {
            'voxel_size': 4,
            'out_channels': 128,
            'emb_dim': DESCRIPTOR_DIM,
            'dropout': 0.5,
            'attn_type': AttentionType.l2attention,
            'nhead': 4
        }
    }
]

model_cfg.discriminator.predictors = [
    {
        'type': LayerType.Predictor,
        'params': {
            'in_channels': 32,
            'out_channels': 64,
            'voxel_size': 8,
            'style_dim': STYLE_DIM
        }
    },
    {
        'type': LayerType.Predictor,
        'params': {
            'in_channels': 64,
            'out_channels': 128,
            'voxel_size': 4,
            'style_dim': STYLE_DIM
        }
    },
    {
        'type': LayerType.Predictor,
        'params': {
            'in_channels': 128,
            'out_channels': 256,
            'voxel_size': 2,
            'style_dim': STYLE_DIM
        }
    }
]