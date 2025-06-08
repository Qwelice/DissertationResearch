import random
from typing import Union, List, Tuple, Optional, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.utils.data import default_collate

from research.data.datasets.modelnet10 import Modelnet10Dataset
from research.modeling.models.discriminator import Discriminator
from research.modeling.models.generator import Generator
from research.utils.constants import OptimizersInitMap
from research.utils.enums import SetType


class MainModule(pl.LightningModule):
    def __init__(self, config):
        super(MainModule, self).__init__()
        self.automatic_optimization = False

        self.config = config
        model_cfg = self.config.model_cfg
        self.generator = Generator(model_cfg)
        self.discriminator = Discriminator(model_cfg)
        self.internal_set = Modelnet10Dataset(config.data_cfg, SetType.train)

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

    def get_optimizer(self, model_params, model_type: Optional[str]=None):
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
        discriminator_opt = self.get_optimizer(self.discriminator.parameters())
        generator_opt = self.get_optimizer(self.generator.parameters())
        return discriminator_opt, generator_opt

    def get_random_image_batch(self, batch_size) -> torch.Tensor:
        indices = random.sample(range(len(self.internal_set)), batch_size)
        samples = [self.internal_set[i] for i in indices]
        batch = default_collate(samples)
        rand_images, _ = batch
        rand_images = rand_images.to(self.device)
        return rand_images

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        dis_opt, gen_opt = self.optimizers()
        image = batch['image']
        voxel = batch['voxel']
        device = image.device
        miss_image = self.get_random_image_batch(self.config.train.batch_size).to(device)

        fakes = self.generator_forward(image)
        fakes_detached = [ f.detach() for f in fakes ]

        # ==================
        # Discriminator part
        # ==================
        dis_opt.zero_grad()
        real_preds = self.discriminator_forward(image, voxel)
        fake_preds = self.discriminator_forward(image, fakes_detached)
        real_preds_miss = self.discriminator_forward(miss_image, voxel)
        fake_preds_miss = self.discriminator_forward(miss_image, fakes_detached)
