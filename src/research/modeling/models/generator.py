from torch import nn
import torch


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

    def _init_layers_(self):
        raise NotImplementedError()