import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def _init_layers_(self):
        raise NotImplementedError()