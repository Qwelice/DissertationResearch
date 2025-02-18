from collections import OrderedDict

import torch
from torch import nn
import torchvision.models as tv_models

from layers.synthesis import SingleDecoderLayer
from layers.synthesis import VoxDecoderLayer
from layers.attention import FeaturesAttentionL2
from src.utils.configuration import Configuration


class SynthesisNetwork(nn.Module):
    def __init__(self):
        super(SynthesisNetwork, self).__init__()
        self.image_tensor = nn.Parameter(torch.zeros(1, 512, 4, 4, 4, dtype=torch.float32))
        self.input_layer = SingleDecoderLayer(in_channels=512,
                                              out_channels=256,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              mod_dim=94,
                                              bank_size=4)

        self.upsampling_1 = VoxDecoderLayer(in_channels=256,
                                            out_channels=128,
                                            voxel_size=4,
                                            patch_size=1,
                                            descriptor_size=256,
                                            mod_dim=94,
                                            bank_size=4,
                                            attn_nhead=4,
                                            attn_emb_dim=512,
                                            xattn_emb_dim=512) # (4, 4, 4) -> (8, 8, 8)

        self.upsampling_2 = VoxDecoderLayer(in_channels=128,
                                            out_channels=64,
                                            voxel_size=8,
                                            patch_size=1,
                                            descriptor_size=256,
                                            mod_dim=94,
                                            bank_size=4,
                                            attn_emb_dim=512,
                                            xattn_emb_dim=512) # (8, 8, 8) -> (16, 16, 16)

        self.output_layer = SingleDecoderLayer(in_channels=64,
                                               out_channels=1,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               mod_dim=94,
                                               bank_size=4,
                                               upsample=True) # (16, 16, 16) -> (32, 32, 32)

    def forward(self, style, descriptor):
        _, ch, d, w, h = self.image_tensor.shape
        image_voxel = self.image_tensor.expand(style.shape[0], ch, d, w, h).contiguous()
        representation, voxel_0 = self.input_layer(image_voxel, style)
        representation, voxel_1 = self.upsampling_1(representation, style, descriptor)
        representation, voxel_2 = self.upsampling_2(representation, style, descriptor)
        _, out = self.output_layer(representation, style)
        return out, voxel_2, voxel_1, voxel_0


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