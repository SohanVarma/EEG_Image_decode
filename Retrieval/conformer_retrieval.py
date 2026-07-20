"""Train the EEG-image retrieval pipeline with a retrieval-specific EEG Conformer.

This module keeps the original training/evaluation code in ``contrast_retrieval.py``
unchanged and registers a new encoder class that is compatible with its model factory.

Run from the Retrieval directory:

    python conformer_retrieval.py \
        --encoder_type EEGConformerRetrievalEncoder \
        --data_path /path/to/Preprocessed_data_250Hz \
        --device cuda:0 \
        --epochs 100 \
        --batch_size 256

Expected EEG input shape: [batch, 63, 250]
Output embedding shape: [batch, 1024]
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import contrast_retrieval as retrieval
from models.loss import ClipLoss


class ConformerBlock(nn.Module):
    """Pre-norm Transformer block used after convolutional EEG tokenization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        expansion: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.attention_norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        x = x + self.attention_dropout(attended)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class AttentionPool(nn.Module):
    """Learn a weighted summary over temporal EEG tokens."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.score = nn.Linear(embed_dim, 1, bias=False)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x + self.query)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class EEGConformerRetrievalEncoder(nn.Module):
    """EEG Conformer adapted for CLIP-space retrieval rather than classification.

    The convolutional front-end extracts local temporal patterns and spatial
    electrode interactions. Transformer blocks then model long-range temporal
    dependencies. A normalized projection head maps the representation into the
    1024-dimensional CLIP image-embedding space expected by this repository.

    ``input_shape`` accepts the tuple passed by the existing generic factory.
    """

    def __init__(
        self,
        input_shape: Sequence[int] | Tuple[int, int] = (63, 250),
        output_dim: int = 1024,
        embed_dim: int = 128,
        num_heads: int = 8,
        depth: int = 4,
        temporal_kernel: int = 25,
        pool_kernel: int = 15,
        pool_stride: int = 5,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        if len(input_shape) != 2:
            raise ValueError("input_shape must be (channels, time_samples)")
        channels, time_samples = int(input_shape[0]), int(input_shape[1])
        if channels <= 0 or time_samples <= 0:
            raise ValueError("input_shape values must be positive")

        self.input_shape = (channels, time_samples)
        self.output_dim = output_dim

        # EEG-Conformer-style convolutional patch embedding:
        # temporal filtering first, followed by a spatial filter spanning all
        # electrodes. The output is a sequence of temporal tokens.
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=(channels, 1),
                groups=embed_dim,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )

        token_count = math.floor((time_samples - pool_kernel) / pool_stride) + 1
        if token_count <= 0:
            raise ValueError("Pooling configuration produces no temporal tokens")

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, token_count, embed_dim)
        )
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.transformer = nn.Sequential(
            *[
                ConformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    expansion=4,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.pool = AttentionPool(embed_dim)

        # Residual projection head follows the projection style already used in
        # this repository, but explicitly normalizes the final CLIP embedding.
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.projection_residual = nn.Linear(embed_dim, output_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, data: Tensor) -> Tensor:
        if data.ndim == 4 and data.shape[1] == 1:
            data = data.squeeze(1)
        if data.ndim != 3:
            raise ValueError(
                f"Expected EEG tensor [batch, channels, time], got {tuple(data.shape)}"
            )
        if tuple(data.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected EEG shape (*, {self.input_shape[0]}, {self.input_shape[1]}), "
                f"got {tuple(data.shape)}"
            )

        x = data.unsqueeze(1)
        x = self.patch_embedding(x)
        x = x.squeeze(2).transpose(1, 2)

        if x.shape[1] != self.positional_embedding.shape[1]:
            raise RuntimeError(
                "Unexpected token count after patch embedding: "
                f"{x.shape[1]} versus {self.positional_embedding.shape[1]}"
            )

        x = x + self.positional_embedding
        x = self.transformer(x)
        pooled = self.pool(x)

        embedding = self.projection(pooled) + self.projection_residual(pooled)
        return F.normalize(embedding, dim=-1)


# Register the encoder in the namespace used by contrast_retrieval.main().
retrieval.EEGConformerRetrievalEncoder = EEGConformerRetrievalEncoder


if __name__ == "__main__":
    retrieval.main()
