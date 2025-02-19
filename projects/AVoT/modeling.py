from collections import OrderedDict

import torch
from torch import nn
import torchvision.models as tv_models

from layers.synthesis import SingleDecoderLayer, VoxDecoderLayer, SingleEncoderLayer, VoxEncoderLayer, DevoxelizingBlock
from layers.attention import FeaturesAttentionL2
from layers.conv import AdaptiveModulatedConv3d
from src.utils.configuration import Configuration


class SynthesisNetwork(nn.Module):
    def __init__(self, mod_dim: int, descriptor_dim: int):
        super(SynthesisNetwork, self).__init__()
        self.image_tensor = nn.Parameter(torch.zeros(1, 512, 4, 4, 4, dtype=torch.float32))
        self.input_layer = SingleDecoderLayer(in_channels=512,
                                              out_channels=256,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              mod_dim=mod_dim,
                                              bank_size=4)

        self.upsampling_1 = VoxDecoderLayer(in_channels=256,
                                            out_channels=128,
                                            voxel_size=4,
                                            patch_size=1,
                                            descriptor_size=descriptor_dim,
                                            mod_dim=mod_dim,
                                            bank_size=4,
                                            attn_nhead=4,
                                            attn_emb_dim=512,
                                            xattn_emb_dim=512) # (4, 4, 4) -> (8, 8, 8)

        self.upsampling_2 = VoxDecoderLayer(in_channels=128,
                                            out_channels=64,
                                            voxel_size=8,
                                            patch_size=1,
                                            descriptor_size=descriptor_dim,
                                            mod_dim=mod_dim,
                                            bank_size=4,
                                            attn_emb_dim=512,
                                            xattn_emb_dim=512) # (8, 8, 8) -> (16, 16, 16)

        self.output_layer = SingleDecoderLayer(in_channels=64,
                                               out_channels=1,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               mod_dim=mod_dim,
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
        self._negative_slope = 0.2
        self.fc_1 = nn.Linear(latent_dim + descriptor_dim, 32)
        self.fc_2 = nn.Linear(32, 64)
        self.fc_3 = nn.Linear(64, 128)
        self.fc_4 = nn.Linear(128, 256)
        self.fc_5 = nn.Linear(256, 256)
        self.fc_6 = nn.Linear(256, 128)

    def forward(self, x):
        device = x.device
        if x.ndim == 3:
            x = x.squeeze(1)
        bs, _ = x.shape
        latent = torch.randn(bs, self._latent_dim, device=device)
        out = torch.cat([x, latent], dim=1)
        out = nn.functional.leaky_relu(self.fc_1(out), self._negative_slope)
        out = nn.functional.leaky_relu(self.fc_2(out), self._negative_slope)
        out = nn.functional.leaky_relu(self.fc_3(out), self._negative_slope)
        out = nn.functional.leaky_relu(self.fc_4(out), self._negative_slope)
        out = nn.functional.leaky_relu(self.fc_5(out), self._negative_slope)
        out = nn.functional.leaky_relu(self.fc_6(out), self._negative_slope)
        return out


class PredictorNetwork(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, voxel_size: int, mod_dim: int):
        super(PredictorNetwork, self).__init__()
        self.conv_1 = AdaptiveModulatedConv3d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              mod_dim=mod_dim,
                                              bank_size=4)
        self.conv_2 = AdaptiveModulatedConv3d(in_channels=out_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              mod_dim=mod_dim,
                                              bank_size=4)
        self.conv_3 = AdaptiveModulatedConv3d(in_channels=out_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              mod_dim=mod_dim,
                                              bank_size=4)
        self.conv_4 = AdaptiveModulatedConv3d(in_channels=out_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              mod_dim=mod_dim,
                                              bank_size=4)
        self.residual = nn.Conv3d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=1)
        self.classifier = nn.Linear(out_channels * voxel_size ** 3, 1)
        self.conv_activation = nn.LeakyReLU(0.2)
        self.cls_activation = nn.Sigmoid()

    def forward(self, x, style):
        x = self.conv_1(x, style)
        x = self.conv_activation(x)
        x = self.conv_2(x, style)
        x = self.conv_activation(x)
        x = self.conv_3(x, style)
        x = self.conv_activation(x)
        x = self.conv_4(x, style)
        x = self.conv_activation(x)
        x = x + self.residual(x)
        out = torch.flatten(x, start_dim=1)
        out = self.classifier(out)
        out = self.cls_activation(out)
        return out


class DiscriminatorNet(nn.Module):
    def __init__(self, descriptor_dim: int):
        super(DiscriminatorNet, self).__init__()
        self.devoxelizers = [
            DevoxelizingBlock(out_channels=32),
            DevoxelizingBlock(out_channels=64),
            DevoxelizingBlock(out_channels=128)
        ]
        self.layers = [
            SingleEncoderLayer(in_channels=32,
                               out_channels=64,
                               downsample=True),  # (32, 32, 32) -> (16, 16, 16)
            VoxEncoderLayer(in_channels=64,
                            out_channels=128,
                            voxel_size_after=8,
                            patch_size_after=1,
                            attn_head=4,
                            attn_emb_dim=512),  # (16, 16, 16) -> (8, 8, 8)
            SingleEncoderLayer(in_channels=128,
                               out_channels=256,
                               downsample=True) # (8, 8, 8) -> (4, 4, 4)
        ]
        self.predictors = [
            PredictorNetwork(in_channels=64,
                             out_channels=32,
                             voxel_size=16,
                             mod_dim=descriptor_dim),
            PredictorNetwork(in_channels=128,
                             out_channels=64,
                             voxel_size=8,
                             mod_dim=descriptor_dim),
            PredictorNetwork(in_channels=256,
                             out_channels=128,
                             voxel_size=4,
                             mod_dim=descriptor_dim)
        ]

    def forward(self, x, style):
        outs = []
        if style.ndim == 3:
            style = style.squeeze(1)
        N = len(self.layers)
        if len(x) - 1 != N:
            raise ValueError('the number of inputs must be equal to the number of layers')
        for i in range(N):
            devox = self.devoxelizers[i]
            inp = devox(x[i]) if devox is not None else x[i]
            temp = self.layers[i](inp)
            preds = [self.predictors[i](temp[0], style)]
            out = (temp[1], preds)
            if i + 1 <= N - 1:
                for j in range(i + 1, N):
                    temp = self.layers[j](temp[0])
                    out[1].append(self.predictors[j](temp[0], style))
            outs.append(out)

        return outs


def model_testing():
    image = torch.empty(8, 3, 224, 224)
    gen_ext = DescriptorFormer(nhead=4,
                               input_dim=512,
                               seq_size=49,
                               output_dim=128,
                               emb_dim=256)
    dis_ext = DescriptorFormer(nhead=4,
                               input_dim=512,
                               seq_size=49,
                               output_dim=128,
                               emb_dim=256)
    mapping = MappingNetwork(128, 128)
    gen = SynthesisNetwork(128, 128)
    dis = DiscriminatorNet(128)
    des_gen = gen_ext(image)
    des_dis = dis_ext(image)
    gen_style = mapping(des_gen[1])
    dis_style = des_dis[1]
    voxels = gen(gen_style, des_gen[0])
    vox_preds = dis(voxels, dis_style)
    print('[voxels shapes]\n')
    for i in range(len(voxels)):
        print(f'[{i}]\t', voxels[i].shape)
    print('\n[voxels and preds shapes]\n')
    for i in range(len(vox_preds)):
        item = vox_preds[i]
        preds = item[1]
        for j in range(len(preds)):
            print(f'[{i},{j}]\t', item[0].shape, ', ', preds[j].shape)


if __name__ == '__main__':
    model_testing()