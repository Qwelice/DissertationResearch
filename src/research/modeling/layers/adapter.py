import math
from typing import Optional, Tuple

import torch
from torch import nn

from research.utils.enums import ConversionType


class AdapterLayer(nn.Module):
    def __init__(self, conversion_type: ConversionType, patch_size: int, image_size: Optional[Tuple[int, int]]=None):
        super(AdapterLayer, self).__init__()
        self.conversion_type = conversion_type
        self.patch_size = patch_size
        self.image_size = image_size

    def _split(self, features: torch.Tensor) -> torch.Tensor:
        """Splits incoming features tensor into patches.

        Args:
            features: incoming features tensor sized of (B, C, H, W).

        Returns:
            Tensor of patches with shape (B, N, D).
        """
        B, C, H, W = features.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, 'feature dimensions must be divisible by patch_size'
        patches = features.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        patch_dim = C * self.patch_size * self.patch_size
        patches = patches.view(B, num_patches, patch_dim)
        return patches

    def _merge(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Merges patches back into full image.

        Args:
            patches: tensor of shape (B, N, D), where D = C * patch_size * patch_size.

        Returns:
            Reconstructed image tensor of shape (B, C, H, W).
        """
        B, N, D = patches.shape
        patch_area = self.patch_size * self.patch_size
        assert D % patch_area == 0, "Patch dimension must be divisible by patch area"
        C = D // patch_area
        patches = patches.view(B, N, C, self.patch_size, self.patch_size)

        # Infer grid size if not given
        if self.image_size is None:
            grid_size = int(math.sqrt(N))
            assert grid_size * grid_size == N, "Cannot infer square grid from N"
            H = W = grid_size * self.patch_size
        else:
            H, W = self.image_size
            assert (H * W) // (self.patch_size * self.patch_size) == N, "Mismatch between image size and number of patches"

        Nh = H // self.patch_size
        Nw = W // self.patch_size
        patches = patches.view(B, Nh, Nw, C, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        output = patches.view(B, C, H, W)
        return output

    def forward(self, x):
        if self.conversion_type == ConversionType.split:
            x = self._split(x)
        elif self.conversion_type == ConversionType.merge:
            x = self._merge(x)
        return x