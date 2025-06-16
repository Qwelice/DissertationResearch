import os

from easydict import EasyDict

from research.config.data_config import data_cfg
from research.config.model_config import model_cfg
from research.utils.enums import OptimizerType, WeightsInitType
from research.utils.io import root_dir

ROOT_DIR = root_dir()
LEARNING_RATE = 0.0025


experiment_cfg = EasyDict()
experiment_cfg.seed = 1488
experiment_cfg.set_name = 'modelnet10'
experiment_cfg.logfile_format = '%Y_%m_%d-%H_%M_%S'
experiment_cfg.logdir_format = '%Y_%m_%d'

# Internal logs params
experiment_cfg.logs = EasyDict()
experiment_cfg.logs.min_level = ''

experiment_cfg.logs.visual = EasyDict()
experiment_cfg.logs.visual.tag = 'vis'
experiment_cfg.logs.visual.voxel_tag = 'vis_vox'
experiment_cfg.logs.visual.plot_tag = 'vis_plt'
experiment_cfg.logs.visual.voxel_threshold = 0.5
experiment_cfg.logs.visual.save_dir = os.path.join(ROOT_DIR, 'logs', 'internal', 'visual')

experiment_cfg.logs.textual = EasyDict()
experiment_cfg.logs.textual.tag = 'tex'
experiment_cfg.logs.textual.save_dir = os.path.join(ROOT_DIR, 'logs', 'internal', 'runtime')

# Training params
experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 16
experiment_cfg.train.learning_rate = LEARNING_RATE
experiment_cfg.train.shuffle = True
experiment_cfg.train.num_workers = 3
experiment_cfg.train.drop_last = True
experiment_cfg.train.num_epochs = 15
experiment_cfg.train.warmup_steps = 700
experiment_cfg.train.warmup_max = 600
experiment_cfg.train.accelerator = 'gpu'
experiment_cfg.train.preferred_device = 'cuda'
experiment_cfg.train.log_every_n_steps = 25

# Evaluation params
experiment_cfg.eval = EasyDict()
experiment_cfg.eval.batch_size = 16
experiment_cfg.eval.shuffle = False
experiment_cfg.eval.num_workers = 3
experiment_cfg.eval.drop_last = True

# Optimizer params
experiment_cfg.optimizer = EasyDict()
experiment_cfg.optimizer = {
    'type': OptimizerType.adamw,
    'params': {
        'params': None,
        'betas': [0.0, 0.99],
        'lr': LEARNING_RATE,
        'weight_decay': 1e-5
    }
}

experiment_cfg.generator = EasyDict()
experiment_cfg.discriminator = EasyDict()

experiment_cfg.generator.init_weights = {
    'type': WeightsInitType.normal,
    'params': {
        'std': 1,
        'mean': 0.0
    }
}
experiment_cfg.discriminator.init_weights = {
    'type': WeightsInitType.normal,
    'params': {
        'std': 1,
        'mean': 0.0
    }
}

# Alternative optimizers params
experiment_cfg.generator.optimizer = {
    'type': OptimizerType.adamw,
    'params': {
        'params': None,
        'betas': [0.0, 0.99],
        'lr': LEARNING_RATE,
        'weight_decay': 1e-5
    }
}
experiment_cfg.discriminator.optimizer = {
    'type': OptimizerType.adamw,
    'params': {
        'params': None,
        'betas': [0.0, 0.99],
        'lr': LEARNING_RATE,
        'weight_decay': 1e-5
    }
}

# LR Scheduler


# Tensorflow params
experiment_cfg.tensorflow = EasyDict()
experiment_cfg.tensorflow.log_dir = os.path.join(ROOT_DIR, 'logs', 'api', 'tensorflow')
experiment_cfg.tensorflow.experiment_name = 'GAMMA-modelnet10'

# Checkpoint params
experiment_cfg.ckpt_dir = os.path.join(ROOT_DIR, 'models', 'GAMMA-modelnet10')
experiment_cfg.monitor = 'val_iou'
experiment_cfg.mode = 'max'
experiment_cfg.save_last = True
experiment_cfg.save_top_k = 10

experiment_cfg.data_cfg = data_cfg
experiment_cfg.model_cfg = model_cfg