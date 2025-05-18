import pytorch_lightning as pl

from core import Configuration
from core.schemas import BuildProvider


class WrappedTrainer(pl.Trainer):
    def __init__(self):
        super(WrappedTrainer, self).__init__()

    @property
    def provider(self) -> BuildProvider:
        return Configuration.build()