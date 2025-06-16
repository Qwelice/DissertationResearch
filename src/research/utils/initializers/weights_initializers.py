import torch.nn as nn
import torch.nn.init as init


def init_weights_normal(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, mean=0.0, std=1.0)
        if m.bias is not None:
            init.zeros_(m.bias)

def init_weights_uniform(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def init_weights_xavier_uniform(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def init_weights_xavier_normal(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def init_weights_kaiming_uniform(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

def init_weights_kaiming_normal(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
