from typing import Optional

from torch.utils.data import Dataset

import src.data.datasets.modelnet10  as modelnet10
from src.data.mappers.default import DefaultMapper
from src.data.storage import DataStorage
from src.data.utils import SetName, SetMode
from src.utils.configuration import Configuration


class DataPicker:
    @staticmethod
    def pick(set_name: SetName, mode: SetMode, config: Configuration) -> Optional[Dataset]:
        fn = DataStorage.get(f'{set_name.value}-{mode.value}')
        objs = fn()
        if set_name == SetName.MODELNET10:
            mapper = DefaultMapper()
            dataset = modelnet10.ModelNet10Set(objs=objs, config=config, mapper=mapper)
        else:
            dataset = None
        return dataset