"""Five-phase volume preprocessing.

Ref: Image acquisition and preprocessing (rigid co-registration to the
portal-venous reference, isotropic resampling, per-phase robust percentile
intensity normalisation, crop to the remnant-liver / surgical-bed region).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def robust_normalize(volume: Tensor, low: float = 1.0, high: float = 99.0) -> Tensor:
    flat = volume.reshape(volume.shape[0], -1)
    lo = torch.quantile(flat, low / 100.0, dim=1, keepdim=True)
    hi = torch.quantile(flat, high / 100.0, dim=1, keepdim=True)
    scaled = (flat - lo) / (hi - lo).clamp_min(1.0e-6)
    return scaled.clamp(0.0, 1.0).reshape_as(volume)


def resample(volume: Tensor, size: tuple[int, int, int]) -> Tensor:
    resized = F.interpolate(volume.unsqueeze(0), size=size, mode="trilinear", align_corners=False)
    return resized.squeeze(0)


def crop_to_mask(volume: Tensor, mask: Tensor, margin: int = 2) -> Tensor:
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return volume
    mins = (coords.min(dim=0).values - margin).clamp_min(0)
    maxs = coords.max(dim=0).values + margin + 1
    return volume[
        :,
        mins[0] : maxs[0],
        mins[1] : maxs[1],
        mins[2] : maxs[2],
    ]


def prepare_volume(phases: Tensor, mask: Tensor | None, size: tuple[int, int, int]) -> Tensor:
    volume = robust_normalize(phases)
    if mask is not None:
        volume = crop_to_mask(volume, mask)
    return resample(volume, size)
