from collections import OrderedDict
from typing import Dict

from torch import nn
from torchvision import models as tv_models
from typing_extensions import Optional

from research.utils.enums import LayerType


def get_resnet(config) -> Optional[nn.Module]:
    params: Dict = config['params']
    resnet = None
    if config['type'] == LayerType.ResNet18:
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    if 'requires_grad' in params.keys():
        for p in resnet.parameters():
            p.requires_grad = params['requires_grad']
    if 'drop_last' in params.keys():
        last = params['drop_last']
        resnet = nn.Sequential(OrderedDict(list(resnet.named_children())[:-last]))
    return resnet