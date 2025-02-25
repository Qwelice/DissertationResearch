import math

import torch
from torch import nn

from . import conv as cnv
from . import attention as atten
from .attention import VoxAttentionL2


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
                                                style_dim=mod_dim,
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
    def __init__(self, in_channels, activation: str='sigmoid'):
        super(VoxelizingBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Tanh()

    def forward(self, voxel):
        raw = self.conv(voxel)
        voxel = self.activation(raw)
        return voxel


class DevoxelizingBlock(nn.Module):
    def __init__(self, out_channels: int, num_groups: int=32):
        super(DevoxelizingBlock, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels=1,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv_2 = nn.Conv3d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.num_groups = math.gcd(out_channels, num_groups)
        self.norm = nn.GroupNorm(self.num_groups, out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.conv_2(out)
        return out


class SingleDecoderLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 style_dim: int,
                 bank_size: int=4,
                 upsample: bool=False):
        super(SingleDecoderLayer, self).__init__()
        self.conv = cnv.AdaptiveModulatedConv3d(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                style_dim=style_dim,
                                                bank_size=bank_size)
        self.voxelizing = VoxelizingBlock(in_channels=out_channels)
        self._upsample = upsample

    def forward(self, x, style):
        if self._upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear')
        x = self.conv(x, style)
        out = self.voxelizing(x)
        return x, out


class SingleEncoderLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int=32,
                 downsample: bool=True):
        super(SingleEncoderLayer, self).__init__()
        self.id_conv = nn.Conv3d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.downsample = nn.Conv3d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    stride=2 if downsample else 1,
                                    padding=1)

        self.num_groups = math.gcd(out_channels, num_groups)
        self.norm = nn.GroupNorm(self.num_groups, out_channels)

        self.activation = nn.LeakyReLU(0.2)

        self.voxelizing = VoxelizingBlock(out_channels)

    def forward(self, x):
        x = self.id_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.downsample(x)
        vox = self.voxelizing(x)
        return x, vox


class VoxEncoderLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 voxel_size_after: int,
                 patch_size_after: int,
                 num_groups: int=32,
                 attn_head: int=4,
                 attn_emb_dim: int=256,
                 attn_tie_qk: bool=True):
        super(VoxEncoderLayer, self).__init__()
        self.id_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                    padding=1)
        self.num_groups = math.gcd(out_channels, num_groups)
        self.norm = nn.GroupNorm(self.num_groups, out_channels)
        self.activation = nn.LeakyReLU(0.2)

        self.attn = VoxAttentionL2(in_channels=out_channels,
                                   nhead=attn_head,
                                   voxel_size=voxel_size_after,
                                   patch_size=patch_size_after,
                                   emb_dim=attn_emb_dim,
                                   tie_qk=attn_tie_qk)
        self.voxelizing = VoxelizingBlock(in_channels=out_channels)

    def forward(self, x):
        x = self.id_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.downsample(x)
        bs, c, d, w, h = x.shape
        out = self.attn(x)
        out = out.view(bs, c, d, w, h)
        vox = self.voxelizing(out)
        return out, vox


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