import torch.optim


def adam(**params):
    return torch.optim.Adam(**params)