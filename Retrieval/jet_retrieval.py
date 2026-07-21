"""JET-style EEG encoder for ATM EEG-to-image retrieval.

This module adapts the representation backbone of *Let EEG Models Learn EEG*
(JET, ICML 2026) to the discriminative ATM retrieval setting:

- non-overlapping temporal EEG patches;
- channel-preserving tokens with temporal and channel embeddings;
- Transformer blocks with adaptive LayerNorm conditioning on flow time;
- attention pooling and a normalized 1024-D CLIP projection;
- optional conditional-flow-matching regularization during retrieval training.

The official JET model is an EEG generator. This file does not claim to be a
verbatim replacement of its generation pipeline; it adapts the published JET
backbone and flow objective for EEG-to-CLIP representation learning.

Run from ``Retrieval``:

    python jet_retrieval.py \
        --encoder_type JETRetrievalEncoder \
        --data_path /path/to/Preprocessed_data_250Hz \
        --device cuda:0 --epochs 40 --batch_size 256
"""

from __future__ import annotations

import builtins
import math
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# The upstream ATM file contains runtime Tensor annotations without importing
# Tensor. Supplying it through builtins keeps this experiment importable without
# changing the original baseline file.
builtins.Tensor = Tensor

import contrast_retrieval as retrieval
from models.loss import ClipLoss


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Create the standard continuous-time sinusoidal embedding."""
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
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class JETBlock(nn.Module):
    """JET-style Transformer block with adaptive LayerNorm and residual gates."""

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

        # DiT/JET-style identity initialization: the conditioning path starts
        # with zero residual gates, stabilizing early optimization.
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.condition(condition).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shift_attn, scale_attn)
        attended, _ = self.attention(h, h, h, need_weights=False)
        x = x + gate_attn[:, None, :] * attended

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(h)
        return x


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm(tokens)
        weights = torch.softmax(self.score(normalized).squeeze(-1), dim=-1)
        return torch.sum(normalized * weights.unsqueeze(-1), dim=1)


class JETRetrievalEncoder(nn.Module):
    """Channel-preserving JET backbone adapted to CLIP-space retrieval."""

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

        # JET preserves channel identity: each channel/temporal patch is a token.
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
                JETBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
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

        # Velocity head predicts the flow from noise to EEG patches.
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, patch_size),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))
        self.loss_func = ClipLoss()

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
        tokens = self.encode_tokens(patches, t)
        pooled = self.pool(tokens)
        embedding = self.projection(pooled) + self.projection_skip(pooled)
        return F.normalize(embedding, dim=-1)

    def flow_matching_loss(self, eeg: Tensor) -> Tensor:
        """Conditional flow matching between Gaussian noise and real EEG."""
        clean = self.patchify(eeg)
        noise = torch.randn_like(clean)
        batch = clean.shape[0]
        t = torch.rand(batch, device=clean.device, dtype=clean.dtype)
        t_view = t[:, None, None, None]
        interpolated = (1.0 - t_view) * noise + t_view * clean
        target_velocity = clean - noise

        tokens = self.encode_tokens(interpolated, t)
        predicted = self.velocity_head(tokens)
        predicted = predicted.view(
            batch, self.input_shape[0], self.patch_count, self.patch_size
        )
        return F.mse_loss(predicted, target_velocity)


# Register with the existing ATM factory.
retrieval.JETRetrievalEncoder = JETRetrievalEncoder


# Wrap the original training step so the JET model receives its auxiliary flow
# objective while every baseline retains the unmodified ATM behavior.
_original_train_model = retrieval.train_model


def train_model_with_jet_flow(
    model,
    dataloader,
    optimizer,
    device,
    text_features_all,
    img_features_all,
):
    if not isinstance(model, JETRetrievalEncoder):
        return _original_train_model(
            model,
            dataloader,
            optimizer,
            device,
            text_features_all,
            img_features_all,
        )

    model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all[::10].to(device).float()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        eeg_data, labels, _, text_features, _, img_features = batch
        eeg_data = eeg_data.to(device).float()
        labels = labels.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        eeg_features = model(eeg_data)
        image_loss = model.loss_func(eeg_features, img_features, model.logit_scale)
        text_loss = model.loss_func(eeg_features, text_features, model.logit_scale)
        flow_loss = model.flow_matching_loss(eeg_data)
        loss = 0.99 * image_loss + 0.01 * text_loss + model.flow_weight * flow_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.detach())
        logits = model.logit_scale.exp().clamp(max=100.0) * eeg_features @ img_features_all.T
        predictions = logits.argmax(dim=1)
        total += predictions.numel()
        correct += (predictions == labels).sum().item()

    return total_loss / max(batch_idx + 1, 1), correct / max(total, 1)


retrieval.train_model = train_model_with_jet_flow


if __name__ == "__main__":
    retrieval.main()
