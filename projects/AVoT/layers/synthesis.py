import math
from collections import OrderedDict

import torch
from torch import nn
import torchvision.models as tv_models

from conv import AdaptiveModulatedConv3d
from attention import VoxAttentionL2, CrossVoxAttentionL2, FeaturesAttentionL2


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


class VoxelizingLayer(nn.Module):
    def __init__(self, in_channels):
        super(VoxelizingLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1)

    def forward(self, voxel):
        return self.conv(voxel)


class DescriptorFormer(nn.Module):
    def __init__(self,
                 nhead: int,
                 input_dim: int,
                 seq_size: int,
                 output_dim: int,
                 emb_dim: int,
                 tie_qk: bool=True):
        super(DescriptorFormer, self).__init__()
        self.features_extractor = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
        self.features_extractor = nn.Sequential(OrderedDict(list(
            self.features_extractor.named_children())[:-2]))
        self.attention = FeaturesAttentionL2(input_dim=input_dim,
                                             seq_size=seq_size,
                                             output_dim=output_dim,
                                             nhead=nhead,
                                             emb_dim=emb_dim,
                                             tie_qk=tie_qk)

    def forward(self, x):
        out = self.features_extractor(x)
        bs, ch, h, w = out.shape
        out = out.view(bs, h * w, ch)
        descriptor = self.attention(out)
        t_local = descriptor[...,:-1,:]
        t_global = descriptor[...,-1:, :]
        return t_local, t_global


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim: int, descriptor_dim: int):
        super(MappingNetwork, self).__init__()
        self._latent_dim = latent_dim
        self.fc_1 = nn.Linear(latent_dim + descriptor_dim, 32)
        self.fc_2 = nn.Linear(32, 64)
        self.fc_3 = nn.Linear(64, 128)
        self.fc_4 = nn.Linear(128, 256)
        self.fc_5 = nn.Linear(256, 256)
        self.fc_6 = nn.Linear(256, 128)

    def forward(self, x):
        device = x.device
        neg_slope = 0.2
        x = x.unsqueeze(1)
        bs, _ = x.shape
        latent = torch.randn(bs, self._latent_dim, device=device)
        out = torch.cat([x, latent], dim=1)
        out = nn.functional.leaky_relu(self.fc_1(out), neg_slope)
        out = nn.functional.leaky_relu(self.fc_2(out), neg_slope)
        out = nn.functional.leaky_relu(self.fc_3(out), neg_slope)
        out = nn.functional.leaky_relu(self.fc_4(out), neg_slope)
        out = nn.functional.leaky_relu(self.fc_5(out), neg_slope)
        out = nn.functional.leaky_relu(self.fc_6(out), neg_slope)
        return out