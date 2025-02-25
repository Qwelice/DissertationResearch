import os.path

from src.utils.configuration import Configuration


def get_configuration(config_name: str) -> Configuration:
    config_name = os.path.join(os.path.dirname(__file__), f'{config_name}.yaml')
    config = Configuration.load_cfg(config_name)
    return config


def get_empty_configuration() -> Configuration:
    return Configuration()


def insert_default_modelnet10_settings(config: Configuration) -> Configuration:
    config.DATA.USE_NORM = True
    config.DATA.NORM_RANGE = [0, 1]
    config.DATA.USE_RESIZE = True
    config.DATA.RESIZE_SHAPE = [224, 224]
    config.DATA.USE_VOXEL = True
    config.DATA.VOXEL.USE_REDUCTION = True
    config.DATA.VOXEL.REDUCTION = 'max'
    config.DATA.VOXEL.REDUCTION_RANK = 2
    config.DATA.TRAIN.BATCH_SIZE = 16
    config.DATA.TRAIN.SHUFFLE = True
    config.DATA.TRAIN.NUM_WORKERS = 2
    config.DATA.TEST.BATCH_SIZE = 16
    config.DATA.TEST.SHUFFLE = False
    config.DATA.TEST.NUM_WORKERS = 2
    return config