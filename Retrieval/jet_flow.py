"""Conditional flow-matching utilities for JET EEG pretraining/regularization."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FlowMatchingHead(nn.Module):
    """Predict patch-space velocity for conditional flow matching."""

    def __init__(self, hidden_dim: int, patch_size: int) -> None:
        super().__init__()
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, patch_size),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.velocity_head(tokens)


def conditional_flow_matching_loss(
    clean_patches: Tensor,
    encode_tokens,
    flow_head: FlowMatchingHead,
) -> Tensor:
    """Compute Gaussian-noise-to-EEG conditional flow-matching MSE."""
    noise = torch.randn_like(clean_patches)
    batch = clean_patches.shape[0]
    t = torch.rand(batch, device=clean_patches.device, dtype=clean_patches.dtype)
    t_view = t[:, None, None, None]
    interpolated = (1.0 - t_view) * noise + t_view * clean_patches
    target_velocity = clean_patches - noise

    tokens = encode_tokens(interpolated, t)
    predicted = flow_head(tokens).view_as(clean_patches)
    return F.mse_loss(predicted, target_velocity)
