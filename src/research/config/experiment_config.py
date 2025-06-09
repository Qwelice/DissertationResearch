import os

from easydict import EasyDict

from research.config.data_config import data_cfg
from research.config.model_config import model_cfg
from research.utils.enums import OptimizerType
from research.utils.io import root_dir

ROOT_DIR = root_dir()

experiment_cfg = EasyDict()
experiment_cfg.seed = 1488
experiment_cfg.set_name = 'modelnet10'
experiment_cfg.num_epochs = 10
experiment_cfg.use_warmup = True
experiment_cfg.warmup_steps = 500

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
experiment_cfg.train.batch_size = 32
experiment_cfg.train.learning_rate = 1e-3
experiment_cfg.train.shuffle = True
experiment_cfg.train.num_workers = 3
experiment_cfg.train.drop_last = True

# Evaluation params
experiment_cfg.eval = EasyDict()
experiment_cfg.eval.batch_size = 1
experiment_cfg.eval.shuffle = False
experiment_cfg.eval.num_workers = 3
experiment_cfg.eval.drop_last = False

# Optimizer params
experiment_cfg.optimizer = EasyDict()
experiment_cfg.optimizer = {
    'type': OptimizerType.adam,
    'params': {
        'params': None,
        'betas': [0.9, 0.999]
    }
}

experiment_cfg.generator = EasyDict()
experiment_cfg.discriminator = EasyDict()

# Alternative optimizers params
experiment_cfg.generator.optimizer = {
    'type': OptimizerType.adam,
    'params': {
        'params': None,
        'betas': [0.9, 0.999]
    }
}
experiment_cfg.discriminator.optimizer = {
    'type': OptimizerType.adam,
    'params': {
        'params': None,
        'betas': [0.9, 0.999]
    }
}

# LR Scheduler


# Tensorflow params
experiment_cfg.tensorflow = EasyDict()
experiment_cfg.tensorflow.log_dir = os.path.join(ROOT_DIR, 'logs', 'api', 'tensorflow')
experiment_cfg.tensorflow.experiment_name = 'GAMMA-modelnet10'

# Checkpoint params
experiment_cfg.ckpt_dir = os.path.join(ROOT_DIR, 'models')
experiment_cfg.monitor = 'val_loss'
experiment_cfg.mode = 'min'

experiment_cfg.data_cfg = data_cfg
experiment_cfg.model_cfg = model_cfg