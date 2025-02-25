import os
from typing import Optional
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from projects.AVoT.lightning import AVoTLit
from projects.configs.base import get_configuration, insert_default_modelnet10_settings
from src.data.datasets.modelnet10 import build_modelnet10_loader
from src.data.utils import SetMode
from src.utils.configuration import Configuration


warnings.filterwarnings('ignore', category=UserWarning, module='pytorch_lightning')
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')


out_dir = os.getenv('outdir')


def _build_configuration(config_name: str):
    config = get_configuration(config_name)
    config = insert_default_modelnet10_settings(config)
    return config


def _build_directories(config: Configuration):
    project_dir = os.path.join(out_dir, config.DIRS.ENTRY_DIR)
    logs = os.path.join(project_dir, config.DIRS.LOG_DIR)
    checkpoints = os.path.join(project_dir, config.DIRS.CKPT_DIR)
    images = os.path.join(project_dir, config.DIRS.IMG_DIR)
    os.makedirs(logs, exist_ok=True)
    os.makedirs(checkpoints, exist_ok=True)
    os.makedirs(images, exist_ok=True)


def _build_loaders(config: Configuration):
    train = build_modelnet10_loader(config, SetMode.TRAIN)
    test = build_modelnet10_loader(config, SetMode.TEST)
    return train, test


def _build_model(config: Configuration):
    external_loader = build_modelnet10_loader(config, SetMode.TRAIN)
    model = AVoTLit(config, ext_loader=external_loader)
    return model


def _build_callbacks(config: Configuration):
    ckpt_dir = os.path.join(out_dir, config.DIRS.CKPT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(ckpt_dir,
                              monitor=config.TRAINER.CKPT.MONITOR,
                              save_last=config.TRAINER.CKPT.SAVE_LAST,
                              save_top_k=config.TRAINER.CKPT.SAVE_TOP_K,
                              mode=config.TRAINER.CKPT.MODE)
    return [ckpt_cb]


def _build_logger(config: Configuration):
    log_dir = os.path.join(out_dir, config.DIRS.LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(log_dir, 'AVoT')
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
    _build_directories(config)
    train_loader, test_loader = _build_loaders(config)
    model = _build_model(config)
    trainer = _build_trainer(config)
    trainer.fit(model, train_loader, test_loader, ckpt_path=ckpt_path)