import os
from argparse import Namespace
from datetime import datetime
from typing import Union, Any, Optional, override, List

import PIL.Image
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
import matplotlib

from research.utils.rendering import VoxelRenderer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


class VisualLogger(Logger):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._date = datetime.now()
        self._running_dir = self._date.strftime(config.logdir_format)
        self._experiment_id = datetime.now().strftime(config.logfile_format)
        self._voxel_tag = self.config.logs.visual.voxel_tag
        self.threshold = self.config.logs.visual.voxel_threshold
        self._renderer = VoxelRenderer()

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_hyperparams(self, params: Union[dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> Optional[str]:
        return self.config.logs.visual.tag

    @property
    def version(self) -> Optional[Union[int, str]]:
        return self._experiment_id

    @override
    @property
    def save_dir(self) -> Optional[str]:
        save_dir = os.path.join(self.config.logs.visual.save_dir, f'{self._running_dir}')
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _create_voxel_grid(self, image_batch: torch.Tensor, voxel_batches: list, clamp=True) -> Image.Image:
        """
        Args:
            image_batch: torch.Tensor [B, C, 224, 224]
            voxel_batches: list of torch.Tensor [B, 256, 256, C], where C = 1 or 3
            clamp: whether to clamp image_batch to [0,1]
        Returns:
            PIL.Image: the resulting grid image
        """
        B = image_batch.size(0)

        if clamp:
            image_batch = image_batch.clamp(0, 1)
            voxel_batches = [vb.clamp(0, 1) for vb in voxel_batches]

        image_batch = image_batch.cpu()
        voxel_batches = [vb.cpu() for vb in voxel_batches]

        grid_images = []

        for i in range(B):
            row = []

            # Object image: [C, H, W] -> [H, W, C]
            obj_img = (image_batch[i] * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            obj_pil = Image.fromarray(obj_img)
            row.append(obj_pil)

            # Voxels
            for vb in voxel_batches:
                vox = vb[i]  # [256, 256, C]
                vox_img = (vox * 255).to(torch.uint8).numpy()

                # Handle grayscale C=1 case
                if vox_img.shape[2] == 1:
                    vox_img = np.squeeze(vox_img, axis=2)

                vox_pil = Image.fromarray(vox_img)
                vox_pil = vox_pil.resize((224, 224), resample=Image.Resampling.BILINEAR)
                row.append(vox_pil)

            row_concat = np.concatenate([np.array(img) for img in row], axis=1)
            grid_images.append(row_concat)

        full_grid = np.concatenate(grid_images, axis=0)
        final_img = Image.fromarray(full_grid)
        return final_img

    @rank_zero_only
    def log_voxels(self, voxels: List[torch.Tensor], epoch: int, batch_idx: int, state: str, image: torch.Tensor):
        self._renderer.device = image.device
        voxel_images = self._renderer.render_voxels(voxels)
        grid = self._create_voxel_grid(image, voxel_images)
        save_dir = os.path.join(self.save_dir, state)
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"{self._voxel_tag}_{state}_epoch-{epoch}_batch-{batch_idx}.png")
        grid.save(img_path)