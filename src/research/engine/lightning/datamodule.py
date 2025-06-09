from typing import Optional

import pytorch_lightning as pl
import torch.utils.data as tdt
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from research.data.datasets.modelnet10 import Modelnet10Dataset
from research.utils.enums import SetType


class MainDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(MainDataModule, self).__init__()
        self.config = config

    def _build_dataloader(self, experiment_config, set_type: SetType,
                         is_pyramidal_voxels: Optional[bool] = None) -> tdt.DataLoader:
        set_name = experiment_config.set_name
        if set_name == 'modelnet10':
            dataset = Modelnet10Dataset(experiment_config.data_cfg, set_type, pyramidal_voxels=is_pyramidal_voxels)
            batch_size = experiment_config.train.batch_size if set_type == SetType.train else experiment_config.eval.batch_size
            shuffle = experiment_config.train.shuffle if set_type == SetType.train else experiment_config.eval.shuffle
            num_workers = experiment_config.train.num_workers if set_type == SetType.train else experiment_config.eval.num_workers
            drop_last = experiment_config.train.drop_last if set_type == SetType.train else experiment_config.eval.drop_last

            loader = tdt.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    drop_last=drop_last)
            return loader
        else:
            raise KeyError(f'unknown dataset name: `{set_name}`')

    def setup(self, stage: str) -> None:
        self.trainset = Modelnet10Dataset(self.config.data_cfg, SetType.train, pyramidal_voxels=True)
        self.evalset = Modelnet10Dataset(self.config.data_cfg, SetType.eval, pyramidal_voxels=True)
        # self.innerset = Modelnet10Dataset(self.config.data_cfg, SetType.train, pyramidal_voxels=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        batch_size = self.config.train.batch_size
        shuffle = self.config.train.shuffle
        num_workers = self.config.train.num_workers
        drop_last = self.config.train.drop_last
        loader = tdt.DataLoader(self.trainset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=drop_last)
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        batch_size = self.config.eval.batch_size
        shuffle = self.config.eval.shuffle
        num_workers = self.config.eval.num_workers
        drop_last = self.config.eval.drop_last
        loader = tdt.DataLoader(self.evalset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=drop_last)
        return loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        batch_size = self.config.eval.batch_size
        shuffle = self.config.eval.shuffle
        num_workers = self.config.eval.num_workers
        drop_last = self.config.eval.drop_last
        loader = tdt.DataLoader(self.evalset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=drop_last)
        return loader