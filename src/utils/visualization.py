import open3d as o3d
import torch
from matplotlib import pyplot as plt

from src.structures.mesh import GenericMesh


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


def visualize_tensor(img: torch.Tensor):
    if img.ndim == 4:
        img = img.squeeze(0)
    if img.shape[0] == 3 or img.shape[0] == 1:
        img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()