import os

from easydict import EasyDict

from research.config.data_config import data_cfg
from research.utils.enums import OptimizerType
from research.utils.io import root_dir

ROOT_DIR = root_dir()

experiment_cfg = EasyDict()
experiment_cfg.seed = 1488
experiment_cfg.set_name = 'modelnet10'
experiment_cfg.num_epochs = 20
experiment_cfg.use_warmup = True
experiment_cfg.warmup_steps = 500
experiment_cfg.output_dir = os.path.join(ROOT_DIR, 'outputs')

# Training params
experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 32
experiment_cfg.train.learning_rate = 1e-3
experiment_cfg.train.continue_from_last = False

# Optimizer params
experiment_cfg.optimizer = EasyDict()
experiment_cfg.optimizer.type = OptimizerType.adam
experiment_cfg.optimizer.betas = [0.9, 0.999]

# LR Scheduler

# Tensorflow params
experiment_cfg.tensorflow = EasyDict()
experiment_cfg.tensorflow.log_dir = os.path.join(ROOT_DIR, 'logs', 'api', 'tensorflow')
experiment_cfg.tensorflow.experiment_name = 'GAMMA-modelnet10'
experiment_cfg.tensorflow.version = '0.1.0'

experiment_cfg.data = data_cfg