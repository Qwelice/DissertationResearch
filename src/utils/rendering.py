from enum import Enum
from typing import Union, Optional

import torch
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, RasterizationSettings, SoftPhongShader
from torch import cos, sin

from src.structures.mesh import GenericMesh


class _RotAxis(Enum):
    X=0,
    Y=1,
    Z=2


def _rotate_fov_cam(axis: _RotAxis, angle: float, rad: bool=False,
                    device: Optional[Union[str, torch.device]]=None) -> torch.Tensor:
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
    radian = torch.tensor([angle], dtype=torch.float32, device=device)
    if not rad:
        radian = radian.deg2rad().to(device)

    if axis == _RotAxis.X:
        R = torch.tensor([
            [1, 0, 0],
            [0, cos(radian), -sin(radian)],
            [0, sin(radian), cos(radian)]
        ], dtype=torch.float32, device=device)
    elif axis == _RotAxis.Y:
        R = torch.tensor([
            [cos(radian), 0, sin(radian)],
            [0, 1, 0],
            [-sin(radian), 0, cos(radian)]
        ], dtype=torch.float32, device=device)
    else:
        R = torch.tensor([
            [cos(radian), -sin(radian), 0],
            [sin(radian), cos(radian), 0],
            [0, 0, 1]
        ])

    return R


def _build_renderer(image_size: int, cameras, lights, device):
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0, faces_per_pixel=1)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    return renderer


def render_mesh(mesh: GenericMesh,
                image_size: int,
                rot: Optional[torch.Tensor]=None,
                dist: Optional[torch.Tensor]=None):
    pass