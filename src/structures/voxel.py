from typing import Optional, Union

import torch
from pytorch3d.ops import cubify

import src.structures.mesh as src_mesh


class Voxel:
    def __init__(self, voxel_indices: torch.Tensor, grid_size: int, device: Optional[Union[str, torch.device]]=None) -> None:
        self.indices = voxel_indices
        self.grid_size = grid_size
        self.device = device

    @staticmethod
    def create_from_solid_box(voxel_grid: torch.Tensor, device: Optional[Union[str, torch.device]]=None ) -> 'Voxel':
        if voxel_grid.ndim == 4:
            voxel_grid = voxel_grid.squeeze(0)
        indices = voxel_grid.nonzero()
        grid_size = voxel_grid.shape[0]
        return Voxel(indices, grid_size, device=device)

    def as_solid_grid(self) -> torch.Tensor:
        grid_dim = (self.grid_size, self.grid_size, self.grid_size)
        voxel_grid = torch.zeros(grid_dim, dtype=torch.uint8)

        voxel_grid[self.indices[:, 0], self.indices[:, 1], self.indices[:, 2]] = 1

        return voxel_grid

    def as_mesh(self) -> 'src_mesh.GenericMesh':
        grid = self.as_solid_grid()
        grid.unsqueeze_(0)
        mesh = cubify(grid, thresh=0.1, device=self.device)
        return src_mesh.GenericMesh.create_from_mesh(mesh=mesh, device=self.device)