import os
from typing import List

import open3d as o3d
import torch
from matplotlib import pyplot as plt
import torchvision.utils as tv_utils

from src.structures.mesh import GenericMesh, MeshColor
from src.structures.voxel import Voxel
from src.utils.rendering import Renderizer


def visualize_mesh(mesh: GenericMesh, size: float=1.0):
    if not isinstance(mesh, GenericMesh):
        raise ValueError('cannot visualize not `GenericMesh` object')
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    mesh = mesh.as_open3d()

    bbox = mesh.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)

    oriented_bbox = mesh.get_oriented_bounding_box()
    oriented_bbox.color = (0, 1, 0)

    o3d.visualization.draw_geometries([mesh, frame, bbox, oriented_bbox])


def visualize_and_save_voxels(voxel_tensors: List[torch.Tensor], save_dir: str, name: str):
    device = voxel_tensors[0].device
    renderizer = Renderizer(256)
    renderizer.setup_camera_motion(angles=(60, 15, 0))
    images = []
    for ten in voxel_tensors:
        voxel = Voxel.create_from_solid_box(voxel_grid=ten, device=device)
        mesh = voxel.as_mesh()
        mesh.change_color(MeshColor.GREEN)
        img = renderizer(mesh)
        img = img.permute(2, 0, 1).unsqueeze(0).cpu()
        images.append(img)
    os.makedirs(save_dir, exist_ok=True)
    tv_utils.save_image(images, os.path.join(save_dir, f'{name}.png'))


def visualize_tensor(img: torch.Tensor):
    if img.ndim == 4:
        img = img.squeeze(0)
    if img.shape[0] == 3 or img.shape[0] == 1:
        img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()