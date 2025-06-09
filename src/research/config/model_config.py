from easydict import EasyDict

from research.utils.enums import LayerType, ConversionType, WeightsInitType

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
DESCRIPTOR_DIM = STYLE_DIM
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
            'out_features': 256
        }
    },
    {
        'type': LayerType.ReLU,
        'params': {}
    },
    {
        'type': LayerType.Linear,
        'params': {
            'in_features': 256,
            'out_features': 512
        }
    },
    {
        'type': LayerType.ReLU,
        'params': {}
    },
    {
        'type': LayerType.Linear,
        'params': {
            'in_features': 512,
            'out_features': STYLE_DIM
        }
    }
]


model_cfg.generator = EasyDict()
model_cfg.generator.base_features = {
    'weights_init': WeightsInitType.xavier_normal,
    'weights_init_params': {},
    'shape': (2, 2, 2)
}
model_cfg.generator.layers = [
    {
        'type': LayerType.GeneratorLayer,
        'params': {
            'input_size': 2,
            'patch_size': 1,
            'emb_dim': DESCRIPTOR_DIM
        },
        'layers': [
            {
                'type': LayerType.AdaConv2d,
                'params': {
                    'in_channels': 2,
                    'out_channels': 4,
                    'style_dim': STYLE_DIM,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'bank_size': 4
                },
                'validate': {
                    'input': (2, 4, 4),
                    'output': (4, 4, 4)
                }
            },
            {
                'type': LayerType.VoxelFormer,
                'params': {
                    'input_size': 4,
                    'seq_size': 16,
                    'dim_size': DESCRIPTOR_DIM,
                    'nhead': 8,
                    'dim_feedforward': 2048,
                    'num_layers': 6,
                    'tiq_qk': True
                }
            }
        ],
        'validate': {
            'input': (2, 2, 2),
            'output': (4, 4, 4)
        }
    },
    {
        'type': LayerType.GeneratorLayer,
        'params': {
            'input_size': 4,
            'patch_size': 2
        },
        'layers': [
            {
                'type': LayerType.AdaConv2d,
                'params': {
                    'in_channels': 4,
                    'out_channels': 8,
                    'style_dim': STYLE_DIM,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'bank_size': 4
                },
                'validate': {
                    'input': (4, 8, 8),
                    'output': (8, 8, 8)
                }
            },
            {
                'type': LayerType.SelfAttention,
                'params': {
                    'embed_dim': DESCRIPTOR_DIM,
                    'num_heads': 8,
                    # 'tie_qk': True
                    'batch_first': True
                }
            },
            {
                'type': LayerType.CrossAttention,
                'params': {
                    'embed_dim': DESCRIPTOR_DIM,
                    'num_heads': 8,
                    # 'tie_qk': True
                    'batch_first': True
                }
            },
            {
                'type': LayerType.VoxelFormer,
                'params': {
                    'input_size': 8,
                    'seq_size': 16,
                    'dim_size': DESCRIPTOR_DIM,
                    'nhead': 8,
                    'dim_feedforward': 2048,
                    'num_layers': 6,
                    'tiq_qk': True
                }
            }
        ]
    },
    {
        'type': LayerType.GeneratorLayer,
        'params': {
            'input_size': 8,
            'patch_size': 2
        },
        'layers': [
            {
                'type': LayerType.AdaConv2d,
                'params': {
                    'in_channels': 8,
                    'out_channels': 16,
                    'style_dim': STYLE_DIM,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'bank_size': 4
                },
                'validate': {
                    'input': (8, 16, 16),
                    'output': (16, 16, 16)
                }
            },
            {
                'type': LayerType.SelfAttention,
                'params': {
                    'embed_dim': DESCRIPTOR_DIM,
                    'num_heads': 8,
                    # 'tie_qk': True
                    'batch_first': True
                }
            },
            {
                'type': LayerType.CrossAttention,
                'params': {
                    'embed_dim': DESCRIPTOR_DIM,
                    'num_heads': 8,
                    # 'tie_qk': True
                    'batch_first': True
                }
            },
            {
                'type': LayerType.VoxelFormer,
                'params': {
                    'input_size': 16,
                    'seq_size': 64,
                    'dim_size': DESCRIPTOR_DIM,
                    'nhead': 8,
                    'dim_feedforward': 2048,
                    'num_layers': 6,
                    'tiq_qk': True
                }
            }
        ],
        'validate': {
            'input': (8, 8, 8),
            'output': (16, 16, 16)
        }
    },
    {
        'type': LayerType.GeneratorLayer,
        'params': {
            'input_size': 16,
            'patch_size': 4
        },
        'layers': [
            {
                'type': LayerType.AdaConv2d,
                'params': {
                    'in_channels': 16,
                    'out_channels': 32,
                    'style_dim': STYLE_DIM,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'bank_size': 4
                },
                'validate': {
                    'input': (16, 32, 32),
                    'output': (32, 32, 32)
                }
            },
            {
                'type': LayerType.SelfAttention,
                'params': {
                    'embed_dim': DESCRIPTOR_DIM,
                    'num_heads': 8,
                    # 'tie_qk': True
                    'batch_first': True
                }
            },
            {
                'type': LayerType.CrossAttention,
                'params': {
                    'embed_dim': DESCRIPTOR_DIM,
                    'num_heads': 8,
                    # 'tie_qk': True
                    'batch_first': True
                }
            },
            {
                'type': LayerType.VoxelFormer,
                'params': {
                    'input_size': 32,
                    'seq_size': 64,
                    'dim_size': DESCRIPTOR_DIM,
                    'nhead': 8,
                    'dim_feedforward': 2048,
                    'num_layers': 6,
                    'tiq_qk': True
                }
            }
        ],
        'validate': {
            'input': (16, 16, 16),
            'output': (32, 32, 32)
        }
    }
]

model_cfg.discriminator = EasyDict()
model_cfg.discriminator.layers = [
    {
        'type': LayerType.DiscriminatorLayer,
        'params': {
            'input_size': 32,
            'patch_size': 4
        },
        'layers': [
            {
                'type': LayerType.Conv2d,
                'params': {
                    'in_channels': 32,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'stride': 2,
                    'padding': 1
                }
            }
        ]
    },
    {
        'type': LayerType.DiscriminatorLayer,
        'params': {
            'input_size': 16,
            'patch_size': 2,
            'dim_size': DESCRIPTOR_DIM,
            'nhead': 8,
            'dim_feedforward': 2048,
            'num_layers': 6,
            'tiq_qk': True
        },
        'layers': [
            {
                'type': LayerType.Conv2d,
                'params': {
                    'in_channels': 16,
                    'out_channels': 8,
                    'kernel_size': 3,
                    'stride': 2,
                    'padding': 1
                }
            }
        ]
    },
    {
        'type': LayerType.DiscriminatorLayer,
        'params': {
            'input_size': 8,
            'patch_size': 2,
            'dim_size': DESCRIPTOR_DIM,
            'nhead': 8,
            'dim_feedforward': 2048,
            'num_layers': 6,
            'tiq_qk': True
        },
        'layers': [
            {
                'type': LayerType.Conv2d,
                'params': {
                    'in_channels': 8,
                    'out_channels': 4,
                    'kernel_size': 3,
                    'stride': 2,
                    'padding': 1
                }
            }
        ]
    },
    {
        'type': LayerType.DiscriminatorLayer,
        'params': {
            'input_size': 4,
            'patch_size': 2
        },
        'layers': [
            {
                'type': LayerType.Conv2d,
                'params': {
                    'in_channels': 4,
                    'out_channels': 1,
                    'kernel_size': 4,
                    'stride': 1,
                    'padding': 0
                }
            }
        ]
    }
]

model_cfg.discriminator.predictors = [
    {
        'type': LayerType.Predictor,
        'params': {
            'in_channels': 16,
            'out_channels': 32,
            'voxel_size': 16,
            'style_dim': STYLE_DIM
        }
    },
    {
        'type': LayerType.Predictor,
        'params': {
            'in_channels': 8,
            'out_channels': 16,
            'voxel_size': 8,
            'style_dim': STYLE_DIM
        }
    },
    {
        'type': LayerType.Predictor,
        'params': {
            'in_channels': 4,
            'out_channels': 8,
            'voxel_size': 4,
            'style_dim': STYLE_DIM
        }
    },
    {
        'type': LayerType.Predictor,
        'params': {
            'in_channels': 1,
            'out_channels': 2,
            'voxel_size': 1,
            'style_dim': STYLE_DIM
        }
    }
]