import math
from typing import Optional, Tuple

import torch
from torch.utils import data as tdt

from research.data.datasets.modelnet10 import Modelnet10Dataset
from research.utils.enums import SetType, LayerType


def build_dataloader(experiment_config, set_type: SetType, is_pyramidal_voxels: Optional[bool]=None) -> tdt.DataLoader:
    set_name = experiment_config.set_name
    if set_name == 'modelnet10':
        dataset = Modelnet10Dataset(experiment_config.data_cfg, set_type, pyramidal_voxels=is_pyramidal_voxels)
        batch_size = experiment_config.train.batch_size if set_type == SetType.train else experiment_config.eval.batch_size
        shuffle = experiment_config.train.shuffle if set_type == SetType.train else experiment_config.eval.shuffle
        num_workers = experiment_config.train.num_workers if set_type == SetType.train else experiment_config.eval.num_workers
        drop_last = experiment_config.train.drop_last if set_type == SetType.train else experiment_config.eval.drop_last

        loader = tdt.DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=drop_last)
        return loader
    else:
        raise KeyError(f'unknown dataset name: `{set_name}`')


def split_into_patches(features: torch.Tensor, patch_size: int, to_flatten: bool=True) -> torch.Tensor:
    """Splits incoming features tensor into patches.

    Args:
        features: incoming features tensor sized of (B, C, H, W).
        patch_size: size of each patch (n, n).
        to_flatten: if True returns sequence of flatten patches.

    Returns:
        Tensor of patches with shape (B, C, Nw, Nh, n, n) whether argument to_flatten is False, in the other case (B, N, D).
    """
    B, C, H, W = features.shape
    assert H % patch_size == 0 and W % patch_size == 0, 'feature dimensions must be divisible by patch_size'
    patches = features.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    if not to_flatten:
        return patches
    num_patches = (H // patch_size) * (W // patch_size)
    patch_dim = C * patch_size * patch_size
    patches = patches.view(B, num_patches, patch_dim)
    return patches


def merge_patches(patches: torch.Tensor, patch_size: int, image_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Merges patches back into full image.

    Args:
        patches: tensor of shape (B, N, D), where D = C * patch_size * patch_size.
        patch_size: size of patch (assumes square).
        image_size: (H, W), optional — required if shape can't be inferred.

    Returns:
        Reconstructed image tensor of shape (B, C, H, W).
    """
    B, N, D = patches.shape
    patch_area = patch_size * patch_size
    assert D % patch_area == 0, "Patch dimension must be divisible by patch area"
    C = D // patch_area
    patches = patches.view(B, N, C, patch_size, patch_size)

    # Infer grid size if not given
    if image_size is None:
        grid_size = int(math.sqrt(N))
        assert grid_size * grid_size == N, "Cannot infer square grid from N"
        H = W = grid_size * patch_size
    else:
        H, W = image_size
        assert (H * W) // (patch_size * patch_size) == N, "Mismatch between image size and number of patches"

    Nh = H // patch_size
    Nw = W // patch_size
    patches = patches.view(B, Nh, Nw, C, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    output = patches.view(B, C, H, W)
    return output


def get_1d_sin_cos_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Creates sinusoidal positional embeddings for 1D sequences.

    Args:
        seq_len: length of the sequence (e.g., number of tokens or patches)
        d_model: embedding dimension

    Returns:
        Tensor of shape (seq_len, d_model)
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2)

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (seq_len, d_model)


def get_2d_sin_cos_pos_embed(height: int, width: int, d_model: int):
    """
    Creates sinusoidal positional embeddings for 2D grids (e.g., image patches).

    Args:
        height: number of positions along height (H)
        width: number of positions along width (W)
        d_model: embedding dimension (must be even)

    Returns:
        Tensor of shape (H * W, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for 2D positional encoding")

    pe_h = get_1d_sin_cos_positional_encoding(height, d_model // 2)  # (H, d_model/2)
    pe_w = get_1d_sin_cos_positional_encoding(width, d_model // 2)   # (W, d_model/2)

    pe = torch.zeros(height, width, d_model)
    for i in range(height):
        for j in range(width):
            pe[i, j] = torch.cat([pe_h[i], pe_w[j]], dim=0)

    pe = pe.view(height * width, d_model)  # (H * W, d_model)
    return pe