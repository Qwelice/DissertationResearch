import random
from typing import Union, List, Tuple, Optional
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.utils.data import default_collate

from research.data.datasets.modelnet10 import Modelnet10Dataset
from research.engine.schedulers.cosine_warmup import CosineWarmupScheduler
from research.modeling.losses.multiscale_loss import MultiScaleLoss
from research.modeling.losses.multiscale_mse import MultiScaleMSE
from research.modeling.models.discriminator import Discriminator
from research.modeling.models.generator import Generator
from research.utils.constants import OptimizersInitMap
from research.utils.enums import SetType
from research.utils.metrics import discriminator_accuracy

warnings.filterwarnings('ignore', category=UserWarning, module='pytorch_lightning')


class MainModule(pl.LightningModule):
    def __init__(self, config):
        super(MainModule, self).__init__()
        self.automatic_optimization = False

        self.config = config
        model_cfg = self.config.model_cfg
        self.generator = Generator(model_cfg)
        self.discriminator = Discriminator(model_cfg)
        self.internal_set = Modelnet10Dataset(config.data_cfg, SetType.train)
        self.mscale_loss = MultiScaleLoss()
        self.mse = MultiScaleMSE()

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
        discriminator_scheduler = CosineWarmupScheduler(discriminator_opt,
                                                        self.config.train.warmup_steps,
                                                        self.trainer.estimated_stepping_batches)
        generator_scheduler = CosineWarmupScheduler(generator_opt,
                                                    self.config.train.warmup_steps,
                                                    self.trainer.estimated_stepping_batches)
        return [discriminator_opt, generator_opt], [discriminator_scheduler, generator_scheduler]

    def get_random_image_batch(self, batch_size) -> torch.Tensor:
        indices = random.sample(range(len(self.internal_set)), batch_size)
        samples = [self.internal_set[i] for i in indices]
        batch = default_collate(samples)
        rand_images = batch['image']
        rand_images = rand_images.to(self.device)
        return rand_images

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        dis_opt, gen_opt = self.optimizers()
        dis_sch, gen_sch = self.lr_schedulers()
        image = batch['image']
        voxel = batch['voxel']
        device = image.device
        if voxel[0].ndim == 5:
            voxel = [v.squeeze(1).to(device) for v in voxel]
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
        dis_loss = self.mscale_loss.D_loss(real_preds, fake_preds, real_preds_miss, fake_preds_miss)
        self.manual_backward(dis_loss)
        dis_opt.step()
        dis_sch.step()
        real_dis_acc = discriminator_accuracy(real_preds, is_real=True)
        fake_dis_acc = discriminator_accuracy(fake_preds, is_real=False)
        dis_acc = (real_dis_acc + fake_dis_acc) / 2

        # ==============
        # Generator part
        # ==============
        gen_opt.zero_grad()
        fake_preds = self.discriminator_forward(image, fakes)
        gen_loss = self.mscale_loss.G_loss(fake_preds) + self.mse(voxel, fakes)
        self.manual_backward(gen_loss)
        gen_opt.step()
        gen_sch.step()

        # ====
        # Logs
        # ====
        self.log('train_dis_acc', dis_acc, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train_dis_loss', dis_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_gen_loss', gen_loss.item(), prog_bar=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image = batch['image']
        voxel = batch['voxel']
        if voxel[0].ndim == 5:
            voxel = [v.squeeze(1).to(image.device) for v in voxel]
        device = image.device
        miss_image = self.get_random_image_batch(self.config.eval.batch_size).to(device)

        fakes = self.generator_forward(image)

        # ==================
        # Discriminator part
        # ==================
        real_preds = self.discriminator_forward(image, voxel)
        fake_preds = self.discriminator_forward(image, fakes)
        real_preds_miss = self.discriminator_forward(miss_image, voxel)
        fake_preds_miss = self.discriminator_forward(miss_image, fakes)
        dis_loss = self.mscale_loss.D_loss(real_preds, fake_preds, real_preds_miss, fake_preds_miss)
        real_dis_acc = discriminator_accuracy(real_preds, is_real=True)
        fake_dis_acc = discriminator_accuracy(fake_preds, is_real=False)
        dis_acc = (real_dis_acc + fake_dis_acc) / 2

        # ==============
        # Generator part
        # ==============
        fake_preds = self.discriminator_forward(image, fakes)
        gen_loss = self.mscale_loss.G_loss(fake_preds) + self.mse(voxel, fakes)

        # ====
        # Logs
        # ====
        self.log('val_dis_acc', dis_acc, prog_bar=False, on_step=True, on_epoch=True)
        self.log('val_dis_loss', dis_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_gen_loss', gen_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        if batch_idx == 0 or batch_idx == self.trainer.num_val_batches[0] - 1:
            self.loggers[1].log_voxels(fakes, self.global_step, image)
