import pytorch_lightning as pl

from core import Configuration


class WrappedTrainer(pl.Trainer):
    def __init__(self):
        super(WrappedTrainer, self).__init__()
        self._config = Configuration()