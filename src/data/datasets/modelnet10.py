import ast
import os.path
from os import PathLike
from typing import Union, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.mappers.default import DefaultMapper
from src.data.picker import DataPicker
from src.data.transforms import Normalization, VoxelReduction
from src.data.utils import SetMode, DataType, SetName
from src.structures.mesh import GenericMesh, MeshColor
from src.utils.configuration import Configuration
from src.utils.rendering import Renderizer


def _read_modelnet10_csv_dicts(metafile: Union[str, PathLike]) -> List[Dict]:
    objs = list()

    meta_df = pd.read_csv(metafile)
    pbar = tqdm(meta_df.iterrows(), total=len(meta_df), desc='dicts reading')

    for idx, row in pbar:
        name = row['object_id']
        mode = row['split']
        category = row['class']
        model = row['object_path']
        pbar.set_postfix({
            'category' : category
        })
        obj = {
            'name' : name,
            'mode' : mode,
            'category' : category,
            'model' : model,
            'voxel' : '',
            'images' : ''
        }
        objs.append(obj)

    return objs


def _read_modelnet10_rd_csv_dicts(metafile: Union[str, PathLike]) -> List[Dict]:
    objs = list()

    meta_df = pd.read_csv(metafile)
    pbar = tqdm(meta_df.iterrows(), total=len(meta_df), desc='dicts reading')

    for idx, row in pbar:
        name = row['name']
        mode = row['mode']
        cat = row['category']
        model = row['model']
        images = row['images']
        pbar.set_postfix({
            'category': cat,
            'name': name
        })
        images = ast.literal_eval(images)
        obj = {
            'name' : name,
            'mode' : mode,
            'category' : cat,
            'model' : model,
            'images' : images,
            'voxel' : ''
        }
        objs.append(obj)

    return objs


def _render_modelnet10_shapes(n_renders: int,
                              objs: List[Dict],
                              dataroot: Union[str, PathLike],
                              img_size: int=256,
                              eps: float=1e-1):
    y_angles = torch.linspace(-60, 60, n_renders, dtype=torch.float32).unsqueeze(0)
    z_angles = torch.empty((1, n_renders), dtype=torch.float32).uniform_(-50, 50)
    x_angles = torch.empty((1, n_renders), dtype=torch.float32).uniform_(-eps, eps)
    angles = torch.cat([x_angles, y_angles, z_angles], dim=0)
    angles = angles.t()
    translation = None

    pbar = tqdm(objs, desc='objects rendering')
    rdz = Renderizer(img_size=img_size)
    for obj in pbar:
        images = []
        cat = obj['category']
        name = obj['name']
        mode = obj['mode']
        model_file = os.path.join(dataroot, 'ModelNet10', obj['model'])

        if not os.path.exists(model_file):
            continue

        mesh = GenericMesh.create_from_file(model_file)
        mesh.swap_y_to_z()
        mesh.to_center()
        mesh.normalize_()
        mesh.change_color(MeshColor.GRADIENT)

        metadir = f'images/{cat}/{mode}/{name}'
        out_dir = os.path.join(dataroot, metadir)
        os.makedirs(out_dir, exist_ok=True)

        for idx, angle in enumerate(angles):
            pbar.set_postfix({
                'image idx': idx,
                'category': cat
            })

            rdz.setup_camera_motion(angle.tolist(), translation)
            image = rdz(mesh)

            img_name = f'{name}_{idx}.png'
            images.append(os.path.join(metadir, img_name))
            plt.imsave(os.path.join(out_dir, img_name), image.cpu().numpy())

        obj['images'] = images
    new_meta_df = pd.DataFrame(objs)
    new_meta_df.to_csv(os.path.join(dataroot, 'metadata_modelnet10-rd.csv'), index=False)


def _voxelize_modelnet10_shapes(objs: List[Dict], dataroot: Union[str, PathLike], grid_size: int):
    pbar = tqdm(objs, desc='model voxelizing')
    for obj in pbar:
        model = obj['model']
        name = obj['name']
        cat = obj['category']
        mode = obj['mode']
        model_file = os.path.join(dataroot, 'ModelNet10', model)
        meta_dir = f'voxels/{cat}/{mode}'
        out_dir = os.path.join(dataroot, meta_dir)
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(model_file):
            continue
        mesh = GenericMesh.create_from_file(model_file)
        mesh.swap_y_to_z()
        mesh.to_center()
        mesh.normalize_()
        voxel = mesh.voxelized(grid_size=grid_size)
        voxel_matrix = voxel.as_solid_grid()
        voxel_name = f'{name}.pt'
        meta_file = os.path.join(meta_dir, voxel_name)
        torch.save(voxel_matrix, os.path.join(out_dir, voxel_name))
        obj['voxel'] = meta_file

    new_meta_df = pd.DataFrame(objs)
    new_meta_df.to_csv(os.path.join(dataroot, 'metadata_modelnet10-rd.csv'), index=False)


def load_modelnet10_dicts(dataroot: Union[str, PathLike], set_mode: SetMode) -> List[Dict]:
    metaname = 'metadata_modelnet10-rd.csv'
    meta_df = pd.read_csv(os.path.join(dataroot, metaname))
    objs = list()

    pbar = tqdm(meta_df.iterrows(), total=len(meta_df), desc=f'modelnet10-{set_mode.value} loading')
    for idx, row in pbar:
        mode = SetMode.recognize(row['mode'])
        if mode == SetMode.VALID:
            mode = SetMode.TEST
        if mode != set_mode:
            continue
        name = row['name']
        cat = row['category']
        images = ast.literal_eval(row['images'])
        model = row['model']
        voxel = row['voxel']
        for image in images:
            obj = {
                'name': name,
                'category': cat,
                'image': os.path.join(dataroot, image),
                'model': os.path.join(dataroot, 'ModelNet10', model),
                'voxel': os.path.join(dataroot, voxel)
            }
            objs.append(obj)

    return objs


class ModelNet10Set(Dataset):
    def __init__(self, objs, config, mapper: DefaultMapper, mode: SetMode=SetMode.TRAIN):
        self._objs = objs
        self._mapper = mapper
        self._config = config
        self._mode = mode
        self._prepare_transforms()

    def _prepare_transforms(self):
        import torchvision.transforms.v2 as tf_v2
        self._mapper.add_transform(DataType.IMAGE, tf_v2.ToImage())
        if self._config.DATA.USE_NORM:
            ranges = self._config.DATA.NORM_RANGE
            self._mapper.add_transform(DataType.IMAGE, Normalization(ranges[0], ranges[1]))
        if self._config.DATA.USE_RESIZE:
            self._mapper.add_transform(DataType.IMAGE, tf_v2.Resize(self._config.DATA.RESIZE_SHAPE))
        if self._config.DATA.USE_VOXEL:
            if self._config.DATA.VOXEL.USE_REDUCTION:
                fn = self._config.DATA.VOXEL.REDUCTION
                rank = self._config.DATA.VOXEL.REDUCTION_RANK
                self._mapper.add_transform(DataType.VOXEL, VoxelReduction(fn, rank))

    def _build_items(self, data_dicts: List[Dict[str, Any]]) -> List[Dict]:
        from PIL import Image
        result = list()
        for data in data_dicts:
            obj = dict()
            image = np.array(Image.open(data['image'], 'r').convert('RGB'))
            obj[DataType.IMAGE] = image
            if self._config.DATA.USE_VOXEL:
                voxel: torch.Tensor = torch.load(data['voxel'], weights_only=False)
                voxel.unsqueeze_(0)
                voxel = voxel.to(dtype=torch.float32)
                obj[DataType.VOXEL] = voxel
            result.append(obj)

        return result

    def __getitem__(self, idx: Union[int, slice]) -> Union[Dict, List[Dict]]:
        is_single = isinstance(idx, int)
        data_dicts = self._objs[idx]
        if is_single:
            data_dicts = [data_dicts]
        items = self._build_items(data_dicts)
        mapped = [self._mapper(item) for item in items]
        if is_single:
            return mapped[0]
        else:
            return mapped

    def __len__(self):
        return len(self._objs)


def build_modelnet10_loader(config: Configuration, mode: SetMode):
    dataset = DataPicker.pick(SetName.MODELNET10, mode, config)
    if mode == SetMode.TRAIN:
        loader = DataLoader(dataset=dataset,
                            batch_size=config.DATA.TRAIN.BATCH_SIZE,
                            shuffle=config.DATA.TRAIN.SHUFFLE,
                            num_workers=config.DATA.TRAIN.NUM_WORKERS)
    else:
        loader = DataLoader(dataset=dataset,
                            batch_size=config.DATA.TEST.BATCH_SIZE,
                            shuffle=config.DATA.TEST.SHUFFLE,
                            num_workers=config.DATA.TEST.NUM_WORKERS)
    return loader