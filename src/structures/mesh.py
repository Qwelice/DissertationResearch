from enum import Enum
from os import PathLike
from typing import Union, Optional, TypeVar

import numpy as np
import torch
from open3d.cpu.pybind.geometry import TriangleMesh
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
import open3d as o3d

from src.structures.voxel import Voxel

T = TypeVar('T')


class MeshColor(Enum):
    BLACK = 'black'
    WHITE = 'white'
    GREEN = 'green'
    RED = 'red'
    BLUE = 'blue'
    GRADIENT = 'gradient'


class GenericMesh:
    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, device: Optional[Union[str, torch.device]]=None):
        if device is None:
            self.device = verts.device
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
            verts = verts.to(self.device)
            faces = faces.to(self.device)

        self.verts = verts
        self.faces = faces
        if self.verts.ndim == 3:
            self.verts = self.verts.squeeze(0).to(self.device)
        if self.faces.ndim == 3:
            self.faces = self.faces.squeeze(0).to(self.device)
        self._normalization_scale: Optional[float] = None

        self.color = MeshColor.WHITE

    @property
    def centroid(self) -> torch.Tensor:
        return torch.mean(self.verts, dim=0, dtype=torch.float32).to(self.device)

    @staticmethod
    def create_from_mesh(mesh: Union[Meshes, TriangleMesh], device: Optional[Union[str, torch.device]]=None) -> 'GenericMesh':
        is_py3d = isinstance(mesh, Meshes)
        if is_py3d:
            if device is None:
                device = mesh.device
            elif isinstance(device, str):
                device = torch.device(device)
            verts = mesh.verts_padded()[0]
            faces = mesh.faces_padded()[0]
        else:
            if device is None:
                device = torch.device('cpu')
            elif isinstance(device, str):
                device = torch.device(device)
            verts = torch.tensor(
                np.asarray(mesh.vertices), dtype=torch.float32, device=device)
            faces = torch.tensor(
                np.asarray(mesh.triangles), dtype=torch.float32, device=device)
        return GenericMesh(verts, faces, device)

    @staticmethod
    def create_from_file(filename: Union[str, PathLike], device: Optional[Union[str, torch.device]]=None) -> 'GenericMesh':
        mesh = o3d.io.read_triangle_mesh(filename)
        return GenericMesh.create_from_mesh(mesh, device)

    def to(self: T, device: Union[str, torch.device]) -> T:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.verts = self.verts.to(self.device)
        self.faces = self.faces.to(self.device)
        self.color = self.color.to(self.device)
        return self

    def copy(self: T) -> T:
        copied = GenericMesh(self.verts, self.faces, self.device)
        copied._normalization_scale = self._normalization_scale
        copied.color = self.color
        return copied

    def cpu(self: T) -> T:
        return self.copy().to(torch.device('cpu'))

    def change_color(self, color: Optional[MeshColor]):
        self.color = color

    def _get_color(self) -> Optional[torch.Tensor]:
        clr = None
        if self.color == MeshColor.WHITE:
            clr = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=self.device)
        elif self.color == MeshColor.BLACK:
            clr = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device)
        elif self.color == MeshColor.RED:
            clr = torch.tensor([[0.9, 0.6, 0.6]], dtype=torch.float32, device=self.device)
        elif self.color == MeshColor.GREEN:
            clr = torch.tensor([[0.6, 0.8, 0.6]], dtype=torch.float32, device=self.device)
        elif self.color == MeshColor.BLUE:
            clr = torch.tensor([[0.6, 0.7, 0.9]], dtype=torch.float32, device=self.device)
        elif self.color == MeshColor.GRADIENT:
            verts_min = self.verts.min(dim=0).values
            verts_max = self.verts.max(dim=0).values
            clr = (self.verts - verts_min) / (verts_max - verts_min)
        if clr is not None and self.color is not MeshColor.GRADIENT:
            clr = clr.expand(self.verts.shape[0], -1)
        return clr

    def as_pytorch3d(self) -> Meshes:
        textures = None
        if self.color is not None:
            clr = self._get_color()
            textures = TexturesVertex(verts_features=[clr])
        return Meshes(verts=[self.verts], faces=[self.faces], textures=textures).to(self.device)

    def as_open3d(self) -> TriangleMesh:
        mesh = TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.verts.cpu())
        mesh.triangles = o3d.utility.Vector3iVector(self.faces.cpu())
        if self.color is not None:
            clr = self._get_color().cpu().numpy()
            mesh.vertex_colors = o3d.utility.Vector3dVector(clr)
        return mesh

    def swap_y_to_z(self: T) -> T:
        self.verts = self.verts[:, [0, 2, 1]] * torch.tensor([1, 1, -1], dtype=torch.float32, device=self.device)
        return self

    def normalize_(self: T) -> T:
        scale = torch.max(self.verts, dim=0).values - torch.min(self.verts, dim=0).values
        scale = torch.max(scale)
        self.verts = self.verts / scale
        self._normalization_scale = scale
        return self

    def denormalize_(self: T) -> T:
        if self._normalization_scale is not None:
            self.verts = self.verts * self._normalization_scale
        return self

    def normalize(self: T) -> T:
        mesh = self.copy().normalize_()
        return mesh

    def denormalize(self: T) -> T:
        mesh = self.copy().denormalize_()
        return mesh

    def translate(self: T, translation: torch.Tensor) -> T:
        self.verts = self.verts + translation
        return self

    def to_center(self: T) -> T:
        self.translate(-self.centroid)

    def voxelized(self, grid_size: int) -> Voxel:
        mesh = self.as_open3d()
        voxel_size = 1.0 / grid_size
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(input=mesh, voxel_size=voxel_size)
        indices = list(map(lambda x: x.grid_index, voxel_grid.get_voxels()))
        indices = np.array(indices)
        indices = torch.tensor(indices, dtype=torch.long)
        indices = torch.clamp(indices, 0, grid_size - 1)
        return Voxel(indices, grid_size=grid_size, device=self.device)