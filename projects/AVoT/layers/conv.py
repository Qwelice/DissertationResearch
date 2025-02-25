import torch
from torch import nn


class AdaptiveModulatedConv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 style_dim: int,
                 demodulate: bool = True,
                 bank_size: int=4,
                 padding: int=0,
                 eps: float=1e-8):
        super(AdaptiveModulatedConv3d, self).__init__()
        self._demodulate = demodulate
        self._stride = stride
        self._padding = padding
        self._eps = eps
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._filter_fc = nn.Linear(style_dim, bank_size)
        self._mod_fc = nn.Linear(style_dim, in_channels)
        self._bank = nn.Parameter(torch.randn(bank_size,
                                              out_channels,
                                              in_channels,
                                              kernel_size,
                                              kernel_size,
                                              kernel_size))

    def forward(self, x, w):
        bs, c_in, dep, wid, hgt = x.shape
        filtered = self._filter_fc(w)
        filter_weight = torch.softmax(filtered, dim=-1)
        weighted = torch.einsum('bn,ncoijk->bcoijk', filter_weight, self._bank)
        modulation = self._mod_fc(w).view(bs, 1, c_in, 1, 1, 1)
        modulated_weights = weighted * modulation
        if self._demodulate:
            demodulation = torch.rsqrt(modulated_weights.pow(2).sum([2, 3, 4, 5], keepdim=True) + self._eps)
            modulated_weights = modulated_weights * demodulation

        modulated_weights = modulated_weights.view(bs * self._out_channels,
                                                   c_in,
                                                   self._kernel_size,
                                                   self._kernel_size,
                                                   self._kernel_size)
        x = x.view(1, bs * c_in, dep, wid, hgt)
        out = torch.conv3d(x, modulated_weights, stride=self._stride, padding=self._padding, groups=bs)
        out = out.view(bs, self._out_channels, out.shape[-3], out.shape[-2], out.shape[-1])
        return out