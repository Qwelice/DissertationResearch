import os.path

from easydict import EasyDict

from research.utils.io import root_dir

ROOT_DIR = root_dir()

process_cfg = EasyDict()

process_cfg.modelnet_dir = os.path.join(ROOT_DIR, 'data', 'datasets', 'ModelNet10_v2')
process_cfg.metadata_file = os.path.join(process_cfg.modelnet_dir, 'metadata_modelnet10.csv')
process_cfg.images_dir = os.path.join(process_cfg.modelnet_dir, 'images')
process_cfg.obj_dir = os.path.join(process_cfg.modelnet_dir, 'ModelNet10')
process_cfg.categories = ['toilet']