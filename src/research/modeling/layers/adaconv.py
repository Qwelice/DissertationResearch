import torch
from torch import nn


class AdaptiveConv2d(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int, style_dim: int,
                 kernel_size: int=3, stride: int=1, padding: int=0,
                 bank_size: int=4, eps: float=1e-8):
        super(AdaptiveConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_dim = style_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bank_size = bank_size
        self._eps = eps
        self.bank_fc = nn.Linear(style_dim, bank_size)
        self.modulation_fc = nn.Linear(style_dim, in_channels)
        self._kernels = nn.Parameter(torch.randn(bank_size, out_channels, in_channels,
                                                 kernel_size, kernel_size))

    def forward(self, x, style):
        bs, c_in, h, w = x.shape
        bank_weights = self.bank_fc(style)
        bank_weights = torch.softmax(bank_weights, dim=-1)
        bank_weights = torch.einsum('bn,noeij->boeij', bank_weights, self._kernels) # (B, Co, Ci, K, K)
        modulation = self.modulation_fc(style).view(bs, 1, c_in, 1, 1)
        bank_weights = bank_weights * modulation
        demodulation = torch.rsqrt(bank_weights.pow(2).sum([2, 3, 4], keepdim=True) + self._eps)
        bank_weights = bank_weights * demodulation
        bank_weights = bank_weights.view(bs * self.out_channels, c_in, self.kernel_size, self.kernel_size)
        x = x.view(1, bs * c_in, h, w)
        out = torch.conv2d(x, bank_weights, stride=self.stride, padding=self.padding, groups=bs)
        out = out.view(bs, self.out_channels, out.shape[-2], out.shape[-1])
        return out