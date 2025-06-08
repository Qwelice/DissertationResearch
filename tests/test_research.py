from typing import Tuple


def test_resnet18():
    import torch
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
    import torch
    from research.config.model_config import model_cfg
    from research.modeling.models.discriminator import Discriminator
    from research.modeling.models.generator import Generator

    batch_size = 32
    image = torch.randn(batch_size, 3, 224, 224)
    generator = Generator(model_cfg)
    discriminator = Discriminator(model_cfg)

    gen_descriptor = generator.get_descriptor(image)
    t_global = gen_descriptor[:, -1, :].squeeze(1)
    t_local = gen_descriptor[:, :-1, :]
    style = generator.get_style(t_global)
    voxels: Tuple = generator(style, t_local)
    voxels = list(reversed(voxels))

    dis_descriptor = discriminator.get_descriptor(image)
    t_global = dis_descriptor[:, -1, :].squeeze(1)
    preds = discriminator(voxels, t_global)
    for i in range(len(preds)):
        print(f'predictions #{i}:')
        for p in preds[i]:
            print(f'\t{p.shape}')