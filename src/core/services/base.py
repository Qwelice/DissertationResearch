from abc import ABC, abstractmethod
from typing import Any

from core.engine.pipeline import Pipeline


class BaseService(ABC):
    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        ...