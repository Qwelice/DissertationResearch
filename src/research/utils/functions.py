import torch


def build_dataloader():
    pass


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