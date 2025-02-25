from typing import Optional, Union

import torch
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from typing_extensions import TypeVar


T = TypeVar('T')


class Voxels:
    def __init__(self,
                 voxel_matrices: torch.Tensor,
                 device: Optional[Union[str, torch.device]]=None):
        if voxel_matrices.ndim != 5:
            raise ValueError('incorrect dim, expected: [B, C, D, W, H]')
        self.voxel_matrices = voxel_matrices
        self.device = device if device is not None else torch.device('cpu')
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.voxel_matrices = self.voxel_matrices.squeeze(1).to(self.device)
        self.indices = self.voxel_matrices.nonzero().cpu()

    def to(self: T, device: Optional[torch.device]=None, dtype: Optional[torch.Type]=None) -> T:
        if device is None:
            self.device = torch.device('cpu')
        self.voxel_matrices.to(device=self.device, dtype=dtype)
        return self

    def as_mesh(self) -> Meshes:
        mesh = cubify(self.voxel_matrices, thresh=0.1, device=self.device)
        return mesh