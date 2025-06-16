from typing import List

import torch
from torch import nn

from research.modeling.losses.dice_loss import DiceLoss


class MultiScaleGenLoss(nn.Module):
    def __init__(self, activation: str='mse'):
        super(MultiScaleGenLoss, self).__init__()
        if activation == 'mse':
            self.criterion = nn.MSELoss()
        elif activation == 'mae':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = DiceLoss()

    def forward(self, real_voxels: List[torch.Tensor], fake_voxels: List[torch.Tensor]):
        criterion = 0.0
        for i in range(len(real_voxels)):
            real = real_voxels[i]
            fake = fake_voxels[i]
            criterion = criterion + self.criterion(real, fake)
        criterion = criterion / len(real_voxels)
        return criterion