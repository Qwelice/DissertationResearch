import torch.optim


def adam(**params):
    return torch.optim.Adam(**params)

def adamw(**params):
    return torch.optim.AdamW(**params)

def sgd(**params):
    return torch.optim.SGD(**params)