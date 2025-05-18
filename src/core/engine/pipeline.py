from core import Configuration
from core.schemas import BuildProvider


class Pipeline:
    def __init__(self):
        pass

    @property
    def provider(self) -> BuildProvider:
        return Configuration.build()