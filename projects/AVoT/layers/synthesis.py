import math

import torch
from torch import nn

from . import conv as cnv
from . import attention as atten


class VoxDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 voxel_size: int,
                 patch_size: int,
                 descriptor_size: int=256,
                 mod_dim: int=128,
                 bank_size: int=4,
                 attn_nhead: int=4,
                 attn_emb_dim: int=512,
                 attn_tie_qk: bool=True,
                 xattn_nhead: int=4,
                 xattn_emb_dim: int=512):
        super(VoxDecoderBlock, self).__init__()
        self.conv = cnv.AdaptiveModulatedConv3d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            mod_dim=mod_dim,
                                            bank_size=bank_size)
        self.attention = atten.VoxAttentionL2(in_channels=out_channels,
                                        nhead=attn_nhead,
                                        voxel_size=voxel_size * 2,
                                        patch_size=patch_size * 2,
                                        emb_dim=attn_emb_dim,
                                        tie_qk=attn_tie_qk)
        self.cross_attention = atten.CrossVoxAttentionL2(in_channels=out_channels,
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


class VoxelizingBlock(nn.Module):
    def __init__(self, in_channels):
        super(VoxelizingBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1)

    def forward(self, voxel):
        return self.conv(voxel)


class SingleDecoderLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 mod_dim: int,
                 bank_size: int=4,
                 upsample: bool=False):
        super(SingleDecoderLayer, self).__init__()
        self.conv = cnv.AdaptiveModulatedConv3d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            mod_dim=mod_dim,
                                            bank_size=bank_size)
        self.voxelizing = VoxelizingBlock(in_channels=out_channels)
        self._upsample = upsample

    def forward(self, x, style):
        x = self.conv(x, style)
        if self._upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear')
        out = self.voxelizing(x)
        return x, out


class VoxDecoderLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 voxel_size: int,
                 patch_size: int,
                 descriptor_size: int,
                 mod_dim: int=128,
                 bank_size: int=4,
                 attn_nhead: int=4,
                 attn_emb_dim: int=256,
                 attn_tie_qk: bool=True,
                 xattn_nhead: int=4,
                 xattn_emb_dim: int=256):
        super(VoxDecoderLayer, self).__init__()
        self.decoding_block = VoxDecoderBlock(in_channels=in_channels,
                                              out_channels=out_channels,
                                              voxel_size=voxel_size,
                                              patch_size=patch_size,
                                              descriptor_size=descriptor_size,
                                              mod_dim=mod_dim,
                                              bank_size=bank_size,
                                              attn_nhead=attn_nhead,
                                              attn_emb_dim=attn_emb_dim,
                                              attn_tie_qk=attn_tie_qk,
                                              xattn_nhead=xattn_nhead,
                                              xattn_emb_dim=xattn_emb_dim)
        self.voxelizing = VoxelizingBlock(in_channels=out_channels)

    def forward(self, voxel, style, descriptor):
        out = self.decoding_block(voxel, style, descriptor)
        vox = self.voxelizing(out)
        return out, vox