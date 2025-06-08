import torch
from torch import nn


class MatchAwareLoss(nn.Module):
    def __init__(self):
        super(MatchAwareLoss, self).__init__()

    def forward(self, real_miss_preds, fake_miss_preds):
        log_loss_real = torch.log(1 + torch.exp(real_miss_preds))
        log_loss_fake = torch.log(1 + torch.exp(fake_miss_preds))
        loss = (log_loss_fake + log_loss_real).mean()
        return loss