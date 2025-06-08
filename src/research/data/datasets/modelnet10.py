from typing import Optional, Dict, Union, List, Tuple
from dataclasses import dataclass
import ast
import os

import numpy as np
import torch
from torch import nn

from research.utils.enums import SetType, recognize_set_type, ReductionType
from research.utils.io import read_anno_file
from torch.utils import data as tdt
import torchvision.transforms.v2 as tf_v2

@dataclass
class _ModelNet10Metadata:
    name: str
    category: str
    image: str
    model: str
    voxel: str


class Modelnet10Dataset(tdt.Dataset):
    def __init__(self, data_config, set_type: SetType,
                 image_transforms: Optional[tf_v2.Transform]=None,
                 voxel_transforms: Optional[tf_v2.Transform]=None,
                 pyramidal_voxels: Optional[bool]=None):
        self._data_cfg = data_config
        self._is_pyramidal = True if pyramidal_voxels else False
        self._set_type = set_type if set_type == SetType.train or set_type == SetType.eval else SetType.eval
        if image_transforms is None:
            image_transforms = data_config.transforms.train.image if set_type == SetType.train else data_config.transforms.eval.image
        if voxel_transforms is None:
            voxel_transforms = data_config.transforms.train.voxel if set_type == SetType.train else data_config.transforms.eval.voxel
        self._image_transforms = image_transforms
        self._voxel_transforms = voxel_transforms
        self._cats: Dict[str, int] = {}
        self._metadata: List[_ModelNet10Metadata] = self._read_metadata_()

    def _read_metadata_(self) -> List[_ModelNet10Metadata]:
        from tqdm import tqdm
        annos = read_anno_file(self._data_cfg.current_dir, self._data_cfg.anno_file)
        pbar = tqdm(annos.iterrows(), total=len(annos), desc=f'modelnet10-{self._set_type.name} loading')
        metadata = []
        for idx, row in pbar:
            set_type = recognize_set_type(row['mode'])
            if set_type == SetType.test:
                set_type = SetType.eval
            if self._set_type != set_type:
                continue
            name = row['name']
            cat = row['category']
            images = ast.literal_eval(row['images'])
            model = row['model']
            voxel = row['voxel']
            if cat not in self._cats:
                self._cats[cat] = 0
            for image in images:
                self._cats[cat] += 1
                data = _ModelNet10Metadata(
                    name=name,
                    category=cat,
                    image=os.path.join(self._data_cfg.current_dir, image),
                    model=os.path.join(self._data_cfg.current_dir, 'ModelNet10', model),
                    voxel=os.path.join(self._data_cfg.current_dir, voxel)
                )
                metadata.append(data)
        return metadata

    def _get_pyramidal(self, base_voxel) -> Tuple[torch.Tensor]:
        cfg = self._data_cfg.transforms.reduction
        tp = cfg['type']
        levels = cfg['levels']
        voxel = base_voxel
        outs = [voxel]
        for _ in range(levels - 1):
            if tp == ReductionType.max:
                voxel = nn.functional.max_pool3d(voxel, kernel_size=3, stride=2, padding=1)
            elif tp == ReductionType.avg:
                voxel = nn.functional.avg_pool3d(voxel, kernel_size=3, stride=2, padding=1)
            outs.append(voxel)
        return outs


    def _build_object_(self, metadata: _ModelNet10Metadata) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        from PIL import Image
        obj = dict()
        image = np.array(Image.open(metadata.image, 'r').convert('RGB'))
        obj['image'] = image
        voxel: torch.Tensor = torch.load(metadata.voxel, weights_only=False)
        voxel.unsqueeze_(0)
        voxel = voxel.to(dtype=torch.float32)
        obj['voxel'] = voxel
        return obj

    def _apply_transforms_(self, data_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        image = data_dict['image']
        voxel = data_dict['voxel']
        image = self._image_transforms(image)
        voxel = self._voxel_transforms(voxel)
        data_dict['image'] = image
        data_dict['voxel'] = voxel
        return data_dict

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        data_dict = self._metadata[index]
        obj = self._build_object_(data_dict)
        transformed = self._apply_transforms_(obj)
        if self._is_pyramidal:
            transformed['voxel'] = self._get_pyramidal(transformed['voxel'])
        return transformed