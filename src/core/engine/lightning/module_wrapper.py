import pytorch_lightning as pl

from core import Configuration
from core.schemas import BuildProvider


class WrappedModule(pl.LightningModule):
    def __init__(self):
        super(WrappedModule, self).__init__()

    @property
    def provider(self) -> BuildProvider:
        return Configuration.build()