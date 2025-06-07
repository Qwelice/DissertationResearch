import torch


def test_resnet18():
    from research.modeling.models.common import get_resnet
    from research.utils.enums import LayerType
    resnet = get_resnet({
        'type': LayerType.ResNet18,
        'params': {
            'requires_grad': False,
            'drop_last': 4
        }
    })
    image = torch.randn((1, 3, 224, 224))
    out = resnet(image)
    print(out.shape)


def test_generator_layer():
    import torch
    from research.config.model_config import model_cfg
    from research.modeling.models.generator import Generator

    batch_size = 32
    image = torch.randn(batch_size, 3, 224, 224)
    generator = Generator(model_cfg)
    descriptor = generator.get_descriptor(image)
    t_global = descriptor[:, -1, :].squeeze(1)
    t_local = descriptor[:, :-1, :]
    style = generator.get_style(t_global)
    voxels = generator(style, t_local)
    for i in range(len(voxels)):
        print(f'voxel #{i}: {voxels[i].shape}')

def test_gan():
    from research.config.model_config import model_cfg
    from research.modeling.models.discriminator import Discriminator
    from research.modeling.models.generator import Generator
    discr = Discriminator(model_cfg)
    gen = Generator(model_cfg)