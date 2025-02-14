from enum import Enum


class SetMode(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'

    @staticmethod
    def recognize(mode: str) -> 'SetMode':
        mode = mode.strip().lower()
        if mode.startswith('train'):
            return SetMode.TRAIN
        elif mode.startswith('test'):
            return SetMode.TEST
        elif mode.startswith('valid'):
            return SetMode.VALID
        else:
            raise ValueError(f'unrecognized mode: `{mode}`')


class DataType(Enum):
    IMAGE = 'image'
    VOXEL = 'voxel'