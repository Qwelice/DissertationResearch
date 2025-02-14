import os.path

from src.utils.configuration import Configuration


def get_configuration(config_name: str) -> Configuration:
    config_name = os.path.join(__file__, f'{config_name}.yaml')
    config = Configuration.load_cfg(config_name)
    return config