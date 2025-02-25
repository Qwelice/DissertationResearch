import os

import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.utils.data import DataLoader

from projects.AVoT.modeling import SynthesisNetwork, DiscriminatorNet
from projects.AVoT.losses import GANLoss, MatchAwareLoss, DiceLoss
from src.data.utils import DataType
from src.utils.configuration import Configuration
from src.utils.visualization import visualize_and_save_voxels


class AVoTLit(pl.LightningModule):
    def __init__(self, config: Configuration, ext_loader: DataLoader):
        super(AVoTLit, self).__init__()
        self.gen = SynthesisNetwork(style_dim=128, descriptor_dim=128, nhead=4, emb_dim=512)
        self.dis = DiscriminatorNet(descriptor_dim=128, nhead=4, emb_dim=512)
        self.gan_loss = GANLoss()
        self.match_loss = MatchAwareLoss()
        self.dice_loss = DiceLoss()
        self.automatic_optimization = False
        self.dis_lr = config.MODEL.GENERATOR.LR
        self.gen_lr = config.MODEL.DISCRIMINATOR.LR
        self.levels_num = config.MODEL.LEVELS_NUM
        self.reduction = config.DATA.VOXEL.REDUCTION
        self.img_dir = os.path.join(os.getenv('outdir'), config.DIRS.IMG_DIR)
        self.external = ext_loader
        self._ext_etr = None

    def _build_pyramidal_voxels(self, base_voxel):
        outs = []
        voxel = base_voxel
        for _ in range(self.levels_num):
            outs.append(voxel)
            if self.reduction == 'max':
                voxel = torch.max_pool3d(voxel, kernel_size=3, stride=2, padding=1)
            else:
                voxel = torch.nn.functional.avg_pool3d(voxel, kernel_size=3, stride=2, padding=1)
        return outs

    def configure_optimizers(self) -> OptimizerLRScheduler:
        dis_opt = torch.optim.AdamW(self.dis.parameters(), lr=self.dis_lr)
        gen_opt = torch.optim.AdamW(self.gen.parameters(), lr=self.gen_lr)
        return dis_opt, gen_opt

    def on_train_epoch_start(self) -> None:
        self._ext_etr = enumerate(self.external)

    def on_validation_epoch_start(self) -> None:
        self._ext_etr = enumerate(self.external)

    def _get_random_batch(self):
        _, batch = next(self._ext_etr)
        return batch

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        dis_opt, gen_opt = self.optimizers()
        images = batch[DataType.IMAGE]
        device = images.device
        voxels = batch[DataType.VOXEL]
        miss_batch = self._get_random_batch()
        miss_images = miss_batch[DataType.IMAGE].to(device)

        real_voxels = self._build_pyramidal_voxels(voxels)

        gen_dstrs = self.gen.get_descriptors(images)
        gen_style = self.gen.get_style(gen_dstrs[1])

        dis_dstrs = self.dis.get_descriptors(images)
        dis_miss_dstrs = self.dis.get_descriptors(miss_images)
        dis_style = dis_dstrs[1]
        dis_miss_style = dis_miss_dstrs[1]

        fake_voxels = self.gen(gen_style, gen_dstrs[0])
        fake_voxels_detached = [voxel.detach() for voxel in fake_voxels]
        # ==================
        # Discriminator part
        # ==================
        dis_opt.zero_grad()
        real_preds = self.dis(real_voxels, dis_style)
        fake_preds = self.dis(fake_voxels_detached, dis_style)
        real_miss_preds = self.dis(real_voxels, dis_miss_style)
        fake_miss_preds = self.dis(fake_voxels_detached, dis_miss_style)
        dis_loss = 0.0
        for i in range(len(real_preds)):
            real_pred = real_preds[i]
            fake_pred = fake_preds[i]
            real_miss_pred = real_miss_preds[i]
            fake_miss_pred = fake_miss_preds[i]
            for j in range(len(real_pred)):
                real = real_pred[j]
                fake = fake_pred[j]
                real_miss = real_miss_pred[j]
                fake_miss = fake_miss_pred[j]
                dis_loss = dis_loss + self.gan_loss.D_loss(real, fake) + self.match_loss(real_miss, fake_miss)

        self.manual_backward(dis_loss)
        dis_opt.step()

        # ==============
        # Generator part
        # ==============
        gen_opt.zero_grad()
        fake_preds = self.dis(fake_voxels, dis_style)
        gen_loss = 0
        for i in range(len(fake_preds)):
            preds = fake_preds[i]
            for j in range(len(preds)):
                pred = preds[j]
                gen_loss = gen_loss + self.gan_loss.G_loss(pred)
            fake_voxel = fake_voxels[i]
            real_voxel = real_voxels[i]
            gen_loss = gen_loss + self.dice_loss(fake_voxel, real_voxel)

        self.manual_backward(gen_loss)
        gen_opt.step()

        self.log('train_dis_loss', dis_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_gen_loss', gen_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        if self.trainer.is_last_batch:
            visualize_and_save_voxels(fake_voxels_detached,
                                      os.path.join(self.img_dir, 'train'),
                                      f'epoch={self.trainer.current_epoch}')

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images = batch[DataType.IMAGE]
        device = images.device
        voxels = batch[DataType.VOXEL]
        miss_batch = self._get_random_batch()
        miss_images = miss_batch[DataType.IMAGE].to(device)
        real_voxels = self._build_pyramidal_voxels(voxels)

        gen_dstrs = self.gen.get_descriptors(images)
        gen_style = self.gen.get_style(gen_dstrs[1])

        dis_dstrs = self.dis.get_descriptors(images)
        dis_miss_dstrs = self.dis.get_descriptors(miss_images)
        dis_style = dis_dstrs[1]
        dis_miss_style = dis_miss_dstrs[1]

        fake_voxels = self.gen(gen_style, gen_dstrs[0])
        # ==================
        # Discriminator part
        # ==================
        real_preds = self.dis(real_voxels, dis_style)
        fake_preds = self.dis(fake_voxels, dis_style)
        real_miss_preds = self.dis(real_voxels, dis_miss_style)
        fake_miss_preds = self.dis(fake_voxels, dis_miss_style)
        dis_loss = 0.0
        for i in range(len(real_preds)):
            real_pred = real_preds[i]
            fake_pred = fake_preds[i]
            real_miss_pred = real_miss_preds[i]
            fake_miss_pred = fake_miss_preds[i]
            for j in range(len(real_pred)):
                real = real_pred[j]
                fake = fake_pred[j]
                real_miss = real_miss_pred[j]
                fake_miss = fake_miss_pred[j]
                dis_loss = dis_loss + self.gan_loss.D_loss(real, fake) + self.match_loss(real_miss, fake_miss)

        # ==============
        # Generator part
        # ==============
        fake_preds = self.dis(fake_voxels, dis_style)
        gen_loss = 0
        for i in range(len(fake_preds)):
            preds = fake_preds[i]
            for j in range(len(preds)):
                pred = preds[j]
                gen_loss = gen_loss + self.gan_loss.G_loss(pred)
            fake_voxel = fake_voxels[i]
            real_voxel = real_voxels[i]
            gen_loss = gen_loss + self.dice_loss(fake_voxel, real_voxel)

        self.log('train_dis_loss', dis_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_gen_loss', gen_loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        if batch_idx < 10:
            visualize_and_save_voxels(fake_voxels,
                                      os.path.join(self.img_dir, 'valid'),
                                      f'epoch={self.trainer.current_epoch}-bid={batch_idx}')