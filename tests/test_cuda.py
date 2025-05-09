import torch


def test():
    assert torch.cuda.is_available() == True