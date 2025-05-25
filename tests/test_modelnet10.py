from research.data.datasets.modelnet10 import Modelnet10Dataset
from research.config.data_config import data_cfg
from research.utils.enums import SetType


def test_dataset ():
    ds = Modelnet10Dataset(data_cfg, SetType.train)
    image_key = 'image'
    voxel_key = 'voxel'
    item = ds[0]
    assert image_key in item, f'dataset item does not contain valid image key'
    assert voxel_key in item, f'dataset item does not contain valid voxel key'