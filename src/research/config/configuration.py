import copy
from typing import Optional, Dict, Union

import yaml

from research.utils import PathLike, PathManager


class Configuration(dict):
    def __init__(self, init_dict: Optional[Dict]=None):
        init_dict = {} if init_dict is None else init_dict
        init_dict = self._create_configuration_tree(init_dict)
        super(Configuration, self).__init__(init_dict)

    @classmethod
    def _create_configuration_tree(cls, dictionary: Dict):
        dictionary = copy.deepcopy(dictionary)
        for k, v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k] = cls(init_dict=v)
        return dictionary

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def load_cfg(cls, cfg_file_path: str) -> 'Configuration':
        with open(cfg_file_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def save_cfg(self, cfg_file_path: PathLike) -> None:
        with open(cfg_file_path, 'w') as f:
            yaml.dump(self, f)


def build_config(config_name) -> Configuration:
    config_file = PathManager.join(PathManager.get_local_path(), f"configs/{config_name}")
    if PathManager.exists(f"{config_file}.yaml"):
        config_file = f"{config_file}.yaml"
    else:
        config_file = f"{config_file}.yml"
    return Configuration.load_cfg(config_file)


def validate_config(config: Configuration, schema: dict, path=""):
    if not isinstance(config, dict):
        raise ValueError(f"{path} must be dictionary")

    for key, value in schema.items():
        full_path = f"{path}.{key}" if path else key

        if key not in config:
            raise ValueError(f"the required key is missing: {full_path}")

        if isinstance(value, dict):
            validate_config(config[key], value, full_path)
        else:
            if not isinstance(config[key], value):
                raise ValueError(f"wrong type for {full_path}: expected {value.__name__}, got {type(config[key]).__name__}")
    return True