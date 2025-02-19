import os
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from projects.AVoT.lightning import AVoTLit
from projects.configs.base import get_configuration, insert_default_modelnet10_settings
from src.data.datasets.modelnet10 import build_modelnet10_loader
from src.data.utils import SetMode
from src.utils.configuration import Configuration


def _build_configuration(config_name: str):
    config = get_configuration(config_name)
    config = insert_default_modelnet10_settings(config)
    return config

def _build_loaders(config: Configuration):
    train = build_modelnet10_loader(config, SetMode.TRAIN)
    test = build_modelnet10_loader(config, SetMode.TEST)
    return train, test

def _build_model(config: Configuration):
    model = AVoTLit(config)
    return model

def _build_callbacks(config: Configuration):
    os.makedirs(config.DIRS.CKPT_DIR, exist_ok=True)
    ckpt_cb = ModelCheckpoint(config.DIRS.CKPT_DIR, monitor='val_gen_loss', save_last=True, save_top_k=3, mode='min')
    return [ckpt_cb]

def _build_logger(config: Configuration):
    os.makedirs(config.DIRS.LOG_DIR, exist_ok=True)
    logger = TensorBoardLogger(config.DIRS.LOG_DIR, 'AVoT')
    return logger

def _build_trainer(config: Configuration):
    logger = _build_logger(config)
    cb = _build_callbacks(config)
    trainer = pl.Trainer(accelerator=config.TRAINER.ACCELERATOR,
                         logger=logger,
                         callbacks=cb,
                         max_epochs=config.TRAINER.MAX_EPOCHS,
                         enable_checkpointing=True,
                         enable_progress_bar=True,
                         enable_model_summary=True)
    return trainer

def start(ckpt_path: Optional[str]=None):
    config = _build_configuration('avot_config_v1')
    train_loader, test_loader = _build_loaders(config)
    model = _build_model(config)
    trainer = _build_trainer(config)
    trainer.fit(model, train_loader, test_loader, ckpt_path=ckpt_path)