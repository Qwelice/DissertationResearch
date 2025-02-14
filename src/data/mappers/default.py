from typing import Dict, Any, List

from torch import nn

from src.data.utils import DataType


class DefaultMapper:
    def __init__(self):
        self._transforms: Dict[DataType, List[nn.Module]] = dict()

    def add_transform(self, tp: DataType, transform: nn.Module):
        if tp not in self._transforms.keys():
            self._transforms[tp] = list()
        self._transforms[tp].append(transform)

    def __call__(self, data_dict: Dict[DataType, Any]):
        if DataType.IMAGE in data_dict.keys() and DataType.IMAGE in self._transforms.keys():
            image = data_dict[DataType.IMAGE]
            for t in self._transforms[DataType.IMAGE]:
                image = t(image)
            data_dict[DataType.IMAGE] = image
        if DataType.VOXEL in data_dict.keys() and DataType.VOXEL in self._transforms.keys():
            voxel = data_dict[DataType.VOXEL]
            if voxel.ndim == 3:
                voxel = voxel.unsqueeze(0)
            for t in self._transforms[DataType.VOXEL]:
                voxel = t(voxel)
            data_dict[DataType.VOXEL] = voxel
        return data_dict