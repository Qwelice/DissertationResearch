import math

from torch import nn
import torch


def _l2_distance(queries, keys):
    q = queries.pow(2).sum(dim=-1, keepdim=True)
    k = keys.pow(2).sum(dim=-1, keepdim=True).transpose(-2, -1)
    dist = q + k - 2 * torch.matmul(queries, keys.transpose(-2, -1))
    return dist


class VoxelPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int=4, embed_dim: int=64, norm_layer: bool=True):
        super(VoxelPatchEmbedding, self).__init__()
        self._proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer:
            self._ln = nn.LayerNorm(embed_dim)
        else:
            self._ln = None

    def forward(self, voxel):
        embed = self._proj(voxel)
        embed = torch.flatten(embed, 2).permute(0, 2, 1)
        if self._ln is not None:
            embed = self._ln(embed)
        return embed


class VoxAttentionL2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 nhead: int,
                 voxel_size: int,
                 patch_size: int,
                 emb_dim: int=64,
                 tie_qk: bool=True):
        super(VoxAttentionL2, self).__init__()
        n_patches = (voxel_size // patch_size)**3
        if tie_qk:
            shared = nn.Linear(emb_dim, nhead * emb_dim)
            self._to_queries = shared
            self._to_keys = shared
        else:
            self._to_queries = nn.Linear(emb_dim, nhead * emb_dim)
            self._to_keys = nn.Linear(emb_dim, nhead * emb_dim)
        self._to_values = nn.Linear(emb_dim, nhead * emb_dim)
        self._emb_dim = emb_dim
        self._nhead = nhead
        self._scale = 1.0 / math.sqrt(emb_dim)

        self._vox_embedding = VoxelPatchEmbedding(in_channels=in_channels, patch_size=patch_size, embed_dim=emb_dim)
        self._pos_embedding = nn.Parameter(torch.zeros(1, n_patches, emb_dim))
        self._alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self._out_proj = nn.Linear(nhead * emb_dim, in_channels * patch_size ** 3)

    def forward(self, voxel):
        emb = self._vox_embedding(voxel) + self._pos_embedding
        queries: torch.Tensor = self._to_queries(emb).unflatten(-1, [self._nhead, self._emb_dim]).transpose(1, 2)
        keys: torch.Tensor = self._to_keys(emb).unflatten(-1, [self._nhead, self._emb_dim]).transpose(1, 2)
        values: torch.Tensor = self._to_values(emb).unflatten(-1, [self._nhead, self._emb_dim]).transpose(1, 2)
        dist = _l2_distance(queries, keys)
        qk = torch.softmax(
            self._alpha * torch.exp(-dist) * self._scale,
            dim=-1)
        attn = qk @ values
        attn = attn.transpose(1, 2).flatten(-2)
        attn = self._out_proj(attn)
        return attn


class CrossVoxAttentionL2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 nhead: int,
                 voxel_size: int,
                 patch_size: int,
                 descriptor_size: int,
                 emb_dim: int=64):
        super(CrossVoxAttentionL2, self).__init__()
        n_patches = (voxel_size // patch_size) ** 3
        self._to_queries = nn.Linear(emb_dim, nhead * emb_dim)
        self._to_keys = nn.Linear(emb_dim, nhead * emb_dim)
        self._to_values = nn.Linear(emb_dim, nhead * emb_dim)
        self._emb_dim = emb_dim
        self._nhead = nhead
        self._scale = 1.0 / math.sqrt(emb_dim)

        self._vox_embedding = VoxelPatchEmbedding(in_channels=in_channels, patch_size=patch_size, embed_dim=emb_dim)
        self._pos_embedding = nn.Parameter(torch.zeros(1, n_patches, emb_dim))
        self._des_embedding = nn.Linear(descriptor_size, emb_dim)
        self._alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self._out_proj = nn.Linear(nhead * emb_dim, in_channels * patch_size ** 3)

    def forward(self, vox_features, descriptor):
        vox_emb = self._vox_embedding(vox_features) + self._pos_embedding
        des_emb = self._des_embedding(descriptor)
        queries = self._to_queries(vox_emb).unflatten(-1, [self._nhead, self._emb_dim]).transpose(1, 2)
        keys = self._to_keys(des_emb).unflatten(-1, [self._nhead, self._emb_dim]).transpose(1, 2)
        values = self._to_values(des_emb).unflatten(-1, [self._nhead, self._emb_dim]).transpose(1, 2)
        dist = _l2_distance(queries, keys)
        qk = torch.softmax(
            self._alpha * torch.exp(-dist) * self._scale,
            dim=-1)
        attn = qk @ values
        attn = attn.transpose(1, 2).flatten(-2)
        attn = self._out_proj(attn)
        return attn