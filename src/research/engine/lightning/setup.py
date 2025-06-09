import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from research.engine.lightning.datamodule import MainDataModule
from research.engine.lightning.module import MainModule
from research.utils.loggers.visual_logger import VisualLogger


def setup_callbacks(exp_cfg):
    callbacks = []
    date = datetime.now().strftime(exp_cfg.logdir_format)
    ckpt_dir = os.path.join(exp_cfg.ckpt_dir, date)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                    monitor=exp_cfg.monitor,
                                    mode=exp_cfg.mode,
                                    save_last=exp_cfg.save_last,
                                    save_top_k=exp_cfg.save_top_k)

    callbacks.append(ckpt_callback)
    return callbacks

def setup_loggers(exp_cfg):
    loggers = []
    date = datetime.now()
    save_dir = date.strftime(exp_cfg.logdir_format)
    save_dir = os.path.join(exp_cfg.tensorflow.log_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=save_dir,
                                  version=date.strftime(exp_cfg.logfile_format),
                                  prefix='GAMMA')
    vis_logger = VisualLogger(exp_cfg)

    loggers.append(tb_logger)
    loggers.append(vis_logger)

    return loggers

def setup_datamodule(exp_cfg):
    dm = MainDataModule(exp_cfg)
    return dm

def setup_model(exp_cfg):
    model = MainModule(exp_cfg)
    return model

def setup_trainer(exp_cfg):
    loggers = setup_loggers(exp_cfg)
    callbacks = setup_callbacks(exp_cfg)

    trainer = pl.Trainer(accelerator=exp_cfg.train.accelerator,
                         logger=loggers,
                         callbacks=callbacks,
                         max_epochs=exp_cfg.train.num_epochs,
                         enable_checkpointing=True,
                         enable_progress_bar=True,
                         enable_model_summary=True,
                         log_every_n_steps=exp_cfg.train.log_every_n_steps)
    return trainer

def setup():
    from research.config.experiment_config import experiment_cfg as exp_cfg

    data = setup_datamodule(exp_cfg)
    model = setup_model(exp_cfg)
    trainer = setup_trainer(exp_cfg)
    return data, model, trainer

__all__ = ['setup']