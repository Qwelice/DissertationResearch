from typing import Union

import torch
from torch import nn
from torchvision.transforms.v2 import ToDtype


class Normalization(nn.Module):
    def __init__(self, x_min: Union[float, torch.Tensor], x_max: Union[float, torch.Tensor]):
        super(Normalization, self).__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.scale = ToDtype(dtype=torch.float32, scale=True)

    def forward(self, image):
        scaled = self.scale(image)
        result = scaled * (self.x_max - self.x_min) + self.x_min
        return result


class VoxelReduction(nn.Module):
    def __init__(self, reduction: str='max', rank: int=0):
        super(VoxelReduction, self).__init__()
        if reduction.startswith('max'):
            self._reduction = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        elif reduction.startswith('avg'):
            self._reduction = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        else:
            self._reduction = None
        self._reduction_rank = rank

    def forward(self, voxel):
        for _ in range(self._reduction_rank):
            if self._reduction is not None:
                voxel = self._reduction(voxel)
        return voxel