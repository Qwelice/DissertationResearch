from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float=0.1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, tgt):
        intersection = (pred * tgt).sum()
        loss = 1. - (2. * intersection + self.smooth) / (pred.sum() + tgt.sum() + self.smooth)
        return loss