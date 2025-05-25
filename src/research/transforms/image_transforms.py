from typing import Union

import torch
from torch import nn
from torchvision.transforms import v2 as tf_v2


class Normalize(nn.Module):
    def __init__(self, min_val: Union[float, torch.Tensor], max_val: Union[float, torch.Tensor]):
        super(Normalize, self).__init__()
        self._min_val = min_val
        self._max_val = max_val
        self._scale = tf_v2.ToDtype(torch.float32, True)

    def forward(self, x):
        scaled = self._scale(x)
        scaled = scaled * (self._max_val - self._min_val) + self._min_val
        return scaled