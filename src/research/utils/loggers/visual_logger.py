import os
from argparse import Namespace
from datetime import datetime
from typing import Union, Any, Optional, override, List

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt
import numpy as np
import torch


class VisualLogger(Logger):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._date = datetime.now()
        self._running_dir = self._date.strftime("%Y%m%d")
        self._experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._voxel_tag = self.config.logs.visual.voxel_tag
        self.threshold = self.config.logs.visual.voxel_threshold

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

    @rank_zero_only
    def log_voxels(self, voxels: List[torch.Tensor], step: int):
        for voxel in range(voxels):
            for i in range(min(len(voxel), 4)):
                if voxel.dim() == 5:
                    voxel = voxel.squeeze(1)

                voxel = voxel.detach().cpu().numpy()
                v = voxels[i]
                fig = plt.figure(figsize=(4, 4))
                ax = fig.add_subplot(111, projection='3d')

                filled = v > self.threshold
                x, y, z = np.where(filled)

                ax.scatter(x, y, z, c=z, cmap='viridis', s=10)
                ax.set_axis_off()
                ax.view_init(30, 120)

                img_path = os.path.join(self.save_dir, f"{self._voxel_tag}_{step:06}_{i}.png")
                plt.savefig(img_path)
                plt.close(fig)