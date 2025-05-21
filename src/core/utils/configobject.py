import copy
import os
from os import PathLike
from typing import Optional, Dict, Union

import yaml


class ConfigObject(dict):
    def __init__(self, init_dict: Optional[Dict]=None):
        init_dict = {} if init_dict is None else init_dict
        init_dict = self._create_configuration_tree(init_dict)
        super(ConfigObject, self).__init__(init_dict)

    def get_plain_keys(self):
        for k, v in self.items():
            pass

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
    def load_cfg(cls, cfg_file_path: str) -> 'ConfigObject':
        with open(cfg_file_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def save_cfg(self, cfg_file_path: Union[str, PathLike]) -> None:
        with open(cfg_file_path, 'w') as f:
            yaml.dump(self, f)


def get_yaml_config(config_name: str) -> ConfigObject:
    config_dir = "" # list(find_project_root(__file__).glob('src/**/config'))[0]
    config_name = os.path.join(config_dir, f'{config_name}')
    if os.path.exists(f'{config_name}.yaml'):
        config_name = f'{config_name}.yaml'
    elif os.path.exists(f'{config_name}.yml'):
        config_name = f'{config_name}.yml'
    else:
        raise FileNotFoundError()
    config = ConfigObject.load_cfg(config_name)
    return config