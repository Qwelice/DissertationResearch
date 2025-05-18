import pytorch_lightning as pl


class WrappedModule(pl.LightningModule):
    def __init__(self):
        super(WrappedModule, self).__init__()
        