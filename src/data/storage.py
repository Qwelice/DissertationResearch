from collections.abc import Callable
from typing import Dict, List


class _DataStorage:
    def __init__(self):
        self._fns: Dict[str, Callable[[], List[Dict]]] = dict()

    def register_dataset(self, name: str, picker_fn: Callable):
        self._fns[name] = picker_fn

    def get(self, name: str) -> Callable:
        if name not in self._fns.keys():
            raise ValueError(f'`{name}` is not registered')
        return self._fns[name]

DataStorage = _DataStorage()