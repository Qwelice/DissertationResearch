import os
from typing import Union

from src.data.datasets.modelnet10 import load_modelnet10_dicts
from src.data.utils import SetMode
from src.data.storage import DataStorage

dataroot = os.getenv('datasets')

modelnet10_splits = {
    f'modelnet10-{SetMode.TRAIN.value}' : (os.path.join(dataroot, 'ModelNet10'), SetMode.TRAIN),
    f'modelnet10-{SetMode.TEST.value}' : (os.path.join(dataroot, 'ModelNet10'), SetMode.TEST)
}

def register_modelnet10(set_name: str, root: Union[str, os.PathLike], set_mode: SetMode):
    DataStorage.register_dataset(set_name, lambda: load_modelnet10_dicts(root, set_mode))

for name, (root_dir, mode) in modelnet10_splits.items():
    register_modelnet10(set_name=name, root=root_dir, set_mode=mode)