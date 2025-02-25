import torch
from torch import nn


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.bce = nn.BCELoss()

    def D_loss(self, real_preds, fake_preds):
        real_labels = torch.ones_like(real_preds)
        fake_labels = torch.zeros_like(fake_preds)
        real_loss = self.bce(real_preds, real_labels)
        fake_loss = self.bce(fake_preds, fake_labels)
        return real_loss + fake_loss

    def G_loss(self, fake_preds):
        real_labels = torch.ones_like(fake_preds)
        return self.bce(fake_preds, real_labels)


class MatchAwareLoss(nn.Module):
    def __init__(self):
        super(MatchAwareLoss, self).__init__()

    def forward(self, real_miss_preds, fake_miss_preds):
        log_loss_real = torch.log(1 + torch.exp(real_miss_preds))
        log_loss_fake = torch.log(1 + torch.exp(fake_miss_preds))
        loss = (log_loss_fake + log_loss_real).mean()
        return loss


def main():
    rmp = torch.empty(32, 1).uniform_(0, 1)
    fmp = torch.empty(32, 1).uniform_(0, 1)
    loss_fn = MatchAwareLoss()
    loss = loss_fn(rmp, fmp)
    print(loss.shape)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float=0.1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, tgt):
        intersection = (pred * tgt).sum()
        loss = 1. - (2. * intersection + self.smooth) / (pred.sum() + tgt.sum() + self.smooth)
        return loss