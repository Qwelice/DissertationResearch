import math

import torch
from torch import nn

from conv import AdaptiveModulatedConv3d
from attention import VoxAttentionL2, CrossVoxAttentionL2


class VoxDecoderLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 voxel_size: int,
                 patch_size: int,
                 descriptor_size: int=256,
                 mod_dim: int=128,
                 attn_nhead: int=4,
                 attn_emb_dim: int=512,
                 attn_tie_qk: bool=True,
                 xattn_nhead: int=4,
                 xattn_emb_dim: int=512):
        super(VoxDecoderLayer, self).__init__()
        self.conv = AdaptiveModulatedConv3d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            mod_dim=mod_dim)
        self.attention = VoxAttentionL2(in_channels=out_channels,
                                        nhead=attn_nhead,
                                        voxel_size=voxel_size * 2,
                                        patch_size=patch_size * 2,
                                        emb_dim=attn_emb_dim,
                                        tie_qk=attn_tie_qk)
        self.cross_attention = CrossVoxAttentionL2(in_channels=out_channels,
                                                   nhead=xattn_nhead,
                                                   voxel_size=voxel_size * 2,
                                                   patch_size=patch_size * 2,
                                                   descriptor_size=descriptor_size,
                                                   emb_dim=xattn_emb_dim)

    def forward(self, voxel, style, descriptor):
        voxel = nn.functional.interpolate(voxel, scale_factor=2, mode='trilinear')
        out = self.conv(voxel, style)
        bs, ch, d, w, h = out.shape
        attn = self.attention(out)
        out = attn.view(bs, ch, d, w, h)
        attn = self.cross_attention(out, descriptor)
        out = attn.view(bs, ch, d, w, h)
        return out