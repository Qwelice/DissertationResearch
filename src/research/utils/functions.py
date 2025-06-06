import math
from typing import Optional, Tuple

import torch
from torch.utils import data as tdt

from research.data.datasets.modelnet10 import Modelnet10Dataset
from research.utils.enums import SetType


def build_dataloader(experiment_config, set_type: SetType) -> tdt.DataLoader:
    ...


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


def get_2d_sincos_pos_embed(height: int, width: int, dim: int, temperature: float = 10000.):
    """
    Creates 2D sin-cos positional encodings shaped of (height * width, dim)

    :param height: image height
    :param width: image width
    :param dim: dimension of positional vector (must be even)
    :param temperature: frequency factor (10000.0 as default)
    :return: tensor shaped of (height * width, dim)
    """
    if dim % 4 != 0:
        raise ValueError("dim must be a multiple of 4 (because x and y encodes along sin and cos)")

    y_pos = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # (H, 1)
    x_pos = torch.arange(width, dtype=torch.float32).unsqueeze(0)  # (1, W)
    y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")  # (H, W)

    omega = torch.arange(dim // 4, dtype=torch.float32)
    omega = 1. / (temperature ** (omega / (dim // 4)))

    y_emb = y_grid.flatten().unsqueeze(1) * omega.unsqueeze(0)  # (H*W, dim//4)
    x_emb = x_grid.flatten().unsqueeze(1) * omega.unsqueeze(0)  # (H*W, dim//4)

    pos_emb = torch.cat([
        torch.sin(x_emb), torch.cos(x_emb),
        torch.sin(y_emb), torch.cos(y_emb)
    ], dim=1)  # (H*W, dim)

    return pos_emb  # shape: (height * width, dim)