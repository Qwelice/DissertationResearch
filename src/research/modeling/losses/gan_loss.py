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