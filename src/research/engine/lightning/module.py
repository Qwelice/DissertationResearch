from typing import Union, List, Tuple, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from research.modeling.models.discriminator import Discriminator
from research.modeling.models.generator import Generator
from research.utils.constants import OptimizersInitMap


class MainModule(pl.LightningModule):
    def __init__(self, config):
        super(MainModule, self).__init__()
        self.config = config
        model_cfg = self.config.model_cfg
        self.generator = Generator(model_cfg)
        self.discriminator = Discriminator(model_cfg)

    def generator_forward(self, image: torch.Tensor) -> Tuple[torch.Tensor]:
        descriptor = self.generator.get_descriptor(image)
        t_global = descriptor[:, -1, :].squeeze(1)
        t_local = descriptor[:, :-1, :]
        style = self.generator.get_style(t_global)
        voxels = self.generator(style, t_local)
        voxels = list(reversed(voxels))
        return voxels

    def discriminator_forward(self, image: torch.Tensor, voxels: Union[List, Tuple]) -> List[List[torch.Tensor]]:
        descriptor = self.discriminator.get_descriptor(image)
        t_global = descriptor[:, -1, :].squeeze(1)
        predictions = self.discriminator(voxels, t_global)
        return predictions

    def _get_optimizer(self, model_params, model_type: Optional[str]=None):
        cfg = self.config.optimizer
        if model_type:
            if model_type.lower().startswith('generator'):
                cfg = self.config.generator.optimizer
            elif model_type.lower().startswith('discriminator'):
                cfg = self.config.discriminator.optimizer
        params = cfg['params']
        params['params'] = model_params
        init_fn = OptimizersInitMap[cfg['type']]
        opt = init_fn(**params)
        return opt

    def configure_optimizers(self) -> OptimizerLRScheduler:
        discriminator_opt = self._get_optimizer(self.discriminator.parameters())
        generator_opt = self._get_optimizer(self.generator.parameters())
        return discriminator_opt, generator_opt