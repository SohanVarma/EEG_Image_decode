"""Reusable JET-style building blocks for EEG representation learning."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Embed continuous flow time values with sinusoidal features."""
    if t.ndim != 1:
        raise ValueError("t must have shape [batch]")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    angles = t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaptive LayerNorm modulation."""
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class JETBlock(nn.Module):
    """Transformer block with adaptive LayerNorm and gated residual paths."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.condition = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.condition(condition).chunk(6, dim=-1)
        )
        h = modulate(self.norm1(x), shift_attn, scale_attn)
        attended, _ = self.attention(h, h, h, need_weights=False)
        x = x + gate_attn[:, None, :] * attended
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp[:, None, :] * self.mlp(h)


class AttentionPool(nn.Module):
    """Learn a weighted global representation from EEG tokens."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm(tokens)
        weights = torch.softmax(self.score(normalized).squeeze(-1), dim=-1)
        return torch.sum(normalized * weights.unsqueeze(-1), dim=1)
