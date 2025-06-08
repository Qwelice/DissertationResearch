from typing import List


def test_dataset ():
    from research.data.datasets.modelnet10 import Modelnet10Dataset
    from research.config.data_config import data_cfg
    from research.utils.enums import SetType

    ds = Modelnet10Dataset(data_cfg, SetType.train, pyramidal_voxels=True)
    image_key = 'image'
    voxel_key = 'voxel'
    item = ds[0]
    assert image_key in item, f'dataset item does not contain valid image key'
    assert voxel_key in item, f'dataset item does not contain valid voxel key'


def test_loader():
    from research.config.experiment_config import experiment_cfg
    from research.utils.enums import SetType
    from research.utils.functions import build_dataloader

    loader = build_dataloader(experiment_cfg, SetType.train, True)
    batch = next(iter(loader))
    print(len(batch['voxel']))

    assert isinstance(batch, List), 'batch item is not list'