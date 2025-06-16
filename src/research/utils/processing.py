import os

import pandas as pd
import torch
from math import sin, cos, pi

from matplotlib import pyplot as plt
from tqdm import tqdm

from research.utils.rendering import Renderizer
from research.utils.structures.generic_mesh import MeshColor, GenericMesh


def render_views(mesh: GenericMesh, renderizer: Renderizer, output_dir, filename, n_views=9):
    mesh.change_color(MeshColor.GRADIENT)
    for i in range(n_views):
        theta = 2 * pi * i / n_views
        renderizer.setup_camera_motion([-pi / 3, 0, theta], None, True)
        img = renderizer(mesh)
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
        plt.imsave(os.path.join(output_dir, f'{filename}_{i}.png'), img)


def process_model(off_path, object_id, render_dir, voxel_dir, views=12):
    mesh = GenericMesh.create_from_file(off_path)
    device = torch.device('cuda')
    mesh = mesh.to(device)
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(voxel_dir, exist_ok=True)
    renderizer = Renderizer(256)

    # Render
    render_views(mesh.copy(), renderizer, render_dir, object_id, n_views=views)

def process_metadata(cfg):
    annos = pd.read_csv(cfg.metadata_file)
    pbar = tqdm(annos.iterrows(), total=len(annos), desc=f'modelnet10 processing')
    for idx, row in pbar:
        category = row['class']
        if category not in cfg.categories:
            continue
        mode = row['split']
        obj_path = row['object_path']
        name = row['object_id']
        pbar.set_postfix({
            'category': category,
            'mode': mode,
            'name': name
        })
        obj_path = os.path.join(cfg.obj_dir, obj_path)
        render_dir = os.path.join(cfg.modelnet_dir, 'images', category, mode, name)
        voxel_dir = os.path.join(cfg.modelnet_dir, 'voxels', category, mode, name)
        process_model(obj_path, name, render_dir, voxel_dir, views=6)

def main():
    from research.config.process_config import process_cfg

    process_metadata(process_cfg)


if __name__ == '__main__':
    main()