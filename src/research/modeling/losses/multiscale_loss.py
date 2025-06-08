from torch import nn

from research.modeling.losses.gan_loss import GANLoss
from research.modeling.losses.matchaware_loss import MatchAwareLoss


class MultiScaleLoss(nn.Module):
    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        self.gan_loss = GANLoss()
        self.ma_loss = MatchAwareLoss()

    def D_loss(self, real_preds, fake_preds, real_preds_miss, fake_preds_miss):
        dis_loss = 0.0
        for i in range(len(real_preds)):
            real_pred = real_preds[i]
            fake_pred = fake_preds[i]
            real_pred_miss = real_preds_miss[i]
            fake_pred_miss = fake_preds_miss[i]
            for j in range(len(real_pred)):
                real = real_pred[j]
                fake = fake_pred[j]
                real_miss = real_pred_miss[j]
                fake_miss = fake_pred_miss[j]
                dis_loss = dis_loss + self.gan_loss.D_loss(real, fake) + self.ma_loss(real_miss, fake_miss)
        return dis_loss

    def G_loss(self, fake_preds):
        gen_loss = 0.0
        for i in range(len(fake_preds)):
            fake_pred = fake_preds[i]
            gen_loss = gen_loss + self.gan_loss.G_loss(fake_pred)
        return gen_loss