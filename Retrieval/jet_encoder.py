"""JET-style EEG encoder adapted to ATM EEG-to-CLIP retrieval."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jet_flow import FlowMatchingHead, conditional_flow_matching_loss
from jet_modules import AttentionPool, JETBlock, sinusoidal_time_embedding
from models.loss import ClipLoss


class JETRetrievalEncoder(nn.Module):
    """Channel-preserving JET backbone with a normalized CLIP projection head."""

    def __init__(
        self,
        input_shape: Sequence[int] | Tuple[int, int] = (63, 250),
        output_dim: int = 1024,
        patch_size: int = 25,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        flow_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError("input_shape must be (channels, time_samples)")
        channels, time_samples = map(int, input_shape)
        if time_samples % patch_size != 0:
            raise ValueError("time_samples must be divisible by patch_size")

        self.input_shape = (channels, time_samples)
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.patch_count = time_samples // patch_size
        self.hidden_dim = hidden_dim
        self.flow_weight = flow_weight

        self.patch_projection = nn.Linear(patch_size, hidden_dim)
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, 1, self.patch_count, hidden_dim)
        )
        self.channel_embedding = nn.Parameter(
            torch.zeros(1, channels, 1, hidden_dim)
        )
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                JETBlock(hidden_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.pool = AttentionPool(hidden_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.projection_skip = nn.Linear(hidden_dim, output_dim, bias=False)
        self.flow_head = FlowMatchingHead(hidden_dim, patch_size)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))
        self.loss_func = ClipLoss()

    @property
    def velocity_head(self):
        """Backward-compatible access for existing tests/checkpoints."""
        return self.flow_head.velocity_head

    def patchify(self, eeg: Tensor) -> Tensor:
        if eeg.ndim == 4 and eeg.shape[1] == 1:
            eeg = eeg.squeeze(1)
        if eeg.ndim != 3:
            raise ValueError("Expected EEG [batch, channels, time]")
        if tuple(eeg.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected EEG shape (*, {self.input_shape[0]}, {self.input_shape[1]}), "
                f"got {tuple(eeg.shape)}"
            )
        return eeg.unfold(-1, self.patch_size, self.patch_size)

    def encode_tokens(self, patches: Tensor, t: Tensor) -> Tensor:
        batch, channels, patch_count, _ = patches.shape
        tokens = self.patch_projection(patches)
        tokens = tokens + self.temporal_embedding + self.channel_embedding
        tokens = tokens.reshape(batch, channels * patch_count, self.hidden_dim)
        condition = self.time_mlp(sinusoidal_time_embedding(t, self.hidden_dim))
        for block in self.blocks:
            tokens = block(tokens, condition)
        return tokens

    def forward(self, eeg: Tensor) -> Tensor:
        patches = self.patchify(eeg)
        t = torch.zeros(patches.shape[0], device=patches.device, dtype=patches.dtype)
        pooled = self.pool(self.encode_tokens(patches, t))
        embedding = self.projection(pooled) + self.projection_skip(pooled)
        return F.normalize(embedding, dim=-1)

    def flow_matching_loss(self, eeg: Tensor) -> Tensor:
        return conditional_flow_matching_loss(
            self.patchify(eeg), self.encode_tokens, self.flow_head
        )
