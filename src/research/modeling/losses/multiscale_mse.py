from typing import List

import torch
from torch import nn


class MultiScaleMSE(nn.Module):
    def __init__(self):
        super(MultiScaleMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, real_voxels: List[torch.Tensor], fake_voxels: List[torch.Tensor]):
        mse = 0.0
        for i in range(len(real_voxels)):
            real = real_voxels[i]
            fake = fake_voxels[i]
            mse = mse + self.mse(real, fake)
        return mse