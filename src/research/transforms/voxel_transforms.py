from typing import Union

import torch
from research.utils.enums import ReductionType
from torch import nn
from torchvision.transforms import v2 as tf_v2


class VoxelReduction(nn.Module):
    def __init__(self, reduction: ReductionType=ReductionType.max, rank: int=0):
        super(VoxelReduction, self).__init__()
        if reduction == ReductionType.max:
            self._reduction = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        elif reduction == ReductionType.avg:
            self._reduction = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        self._rank = rank

    def forward(self, x):
        for _ in range(self._rank):
            x = self._reduction(x)
        return x