import os.path

from easydict import EasyDict
from research.transforms.image_transforms import Normalize
from research.transforms.voxel_transforms import VoxelReduction
from torchvision.transforms import v2 as tf_v2

from research.utils.io import root_dir

ROOT_DIR = root_dir()

data_cfg = EasyDict()

data_cfg.set_name = 'modelnet10'
data_cfg.path_to_data = os.path.join(ROOT_DIR, 'data', 'datasets')
data_cfg.current_dir = os.path.join(data_cfg.path_to_data, 'ModelNet10')
data_cfg.anno_file = 'metadata_modelnet10-rd.csv'


data_cfg.transforms = EasyDict()
data_cfg.transforms.train = EasyDict()
data_cfg.transforms.eval = EasyDict()

data_cfg.transforms.train.image = tf_v2.Compose([
    tf_v2.ToImage(),
    Normalize(0, 1),
    tf_v2.Resize(224)
])
data_cfg.transforms.train.voxel = tf_v2.Compose([
    VoxelReduction(rank=3)
])
data_cfg.transforms.eval.image = tf_v2.Compose([
    tf_v2.ToImage(),
    Normalize(0, 1),
    tf_v2.Resize(224)
])
data_cfg.transforms.eval.voxel = tf_v2.Compose([
    VoxelReduction(rank=3)
])

