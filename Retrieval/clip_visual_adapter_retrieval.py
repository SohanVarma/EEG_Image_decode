"""Train EEG retrieval with a frozen-CLIP residual visual adapter.

The dataset continues to provide precomputed 1024-D CLIP image features. This
module learns a small residual adapter on top of those frozen features while
training the EEG Conformer. Text targets remain unchanged.

Run from ``Retrieval``::

    python clip_visual_adapter_retrieval.py \
        --encoder_type EEGConformerVisualAdapterEncoder \
        --data_path /path/to/Preprocessed_data_250Hz \
        --device cuda:0 --epochs 100 --batch_size 256
"""

from __future__ import annotations

import random
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import contrast_retrieval as retrieval
from conformer_retrieval import EEGConformerRetrievalEncoder


class ResidualVisualAdapter(nn.Module):
    """Task-adapt frozen CLIP image features without changing their dimension.

    The final projection is zero-initialized, so the adapter begins as an
    identity mapping. A learnable residual scale further limits early drift
    away from the pretrained CLIP space.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
        initial_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("feature dimensions must be positive")

        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"Expected image features [batch, dim], got {tuple(features.shape)}"
            )
        residual = self.norm(features)
        residual = self.down(residual)
        residual = F.gelu(residual)
        residual = self.dropout(residual)
        residual = self.up(residual)
        adapted = features + self.residual_scale * residual
        return F.normalize(adapted, dim=-1)


class EEGConformerVisualAdapterEncoder(EEGConformerRetrievalEncoder):
    """Retrieval Conformer plus an image-side CLIP residual adapter."""

    def __init__(
        self,
        input_shape: Sequence[int] | Tuple[int, int] = (63, 250),
        output_dim: int = 1024,
        adapter_bottleneck: int = 256,
        adapter_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(input_shape=input_shape, output_dim=output_dim, **kwargs)
        self.visual_adapter = ResidualVisualAdapter(
            feature_dim=output_dim,
            bottleneck_dim=adapter_bottleneck,
            dropout=adapter_dropout,
        )

    def adapt_image_features(self, image_features: Tensor) -> Tensor:
        return self.visual_adapter(image_features)


def train_model_with_visual_adapter(
    model,
    dataloader,
    optimizer,
    device,
    text_features_all,
    img_features_all,
):
    model.train()
    text_features_all = F.normalize(text_features_all.to(device).float(), dim=-1)
    raw_img_features_all = img_features_all[::10].to(device).float()

    total_loss = 0.0
    correct = 0
    total = 0
    alpha = 0.99

    for batch_idx, batch in enumerate(dataloader):
        eeg_data, labels, _, text_features, _, img_features = batch
        eeg_data = eeg_data.to(device)
        labels = labels.to(device)
        text_features = F.normalize(text_features.to(device).float(), dim=-1)
        img_features = img_features.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        eeg_features = model(eeg_data).float()
        adapted_img_features = model.adapt_image_features(img_features)

        img_loss = model.loss_func(
            eeg_features, adapted_img_features, model.logit_scale
        )
        text_loss = model.loss_func(eeg_features, text_features, model.logit_scale)
        loss = alpha * img_loss + (1.0 - alpha) * text_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())

        with torch.no_grad():
            adapted_all = model.adapt_image_features(raw_img_features_all)
            logit_scale = model.logit_scale.exp().clamp(max=100.0)
            logits = logit_scale * eeg_features @ adapted_all.T
            predicted = torch.argmax(logits, dim=1)
            total += predicted.numel()
            correct += (predicted == labels).sum().item()

    return total_loss / (batch_idx + 1), correct / max(total, 1)


def evaluate_model_with_visual_adapter(
    model,
    dataloader,
    device,
    text_features_all,
    img_features_all,
    k,
):
    model.eval()
    text_features_all = F.normalize(text_features_all.to(device).float(), dim=-1)
    raw_img_features_all = img_features_all.to(device).float()

    total_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0
    alpha = 0.99
    all_labels = set(range(text_features_all.size(0)))

    with torch.no_grad():
        adapted_img_features_all = model.adapt_image_features(raw_img_features_all)
        logit_scale = model.logit_scale.exp().clamp(max=100.0)

        for batch_idx, batch in enumerate(dataloader):
            eeg_data, labels, _, text_features, _, img_features = batch
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)
            text_features = F.normalize(text_features.to(device).float(), dim=-1)
            adapted_img_features = model.adapt_image_features(
                img_features.to(device).float()
            )

            eeg_features = model(eeg_data).float()
            img_loss = model.loss_func(
                eeg_features, adapted_img_features, model.logit_scale
            )
            text_loss = model.loss_func(
                eeg_features, text_features, model.logit_scale
            )
            total_loss += float((alpha * img_loss + (1 - alpha) * text_loss).item())

            for idx, label in enumerate(labels):
                label_value = label.item()
                possible = list(all_labels - {label_value})
                selected_classes = random.sample(possible, k - 1) + [label_value]
                selected_features = adapted_img_features_all[selected_classes]
                logits = logit_scale * eeg_features[idx] @ selected_features.T
                ranking = torch.argsort(logits, descending=True)
                predicted_label = selected_classes[ranking[0].item()]
                correct += int(predicted_label == label_value)
                total += 1

                if k == 200:
                    top_indices = ranking[:5].tolist()
                    top_labels = [selected_classes[i] for i in top_indices]
                    top5_correct += int(label_value in top_labels)

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / max(total, 1)
    top5_accuracy = top5_correct / max(total, 1) if k == 200 else 0.0
    return average_loss, accuracy, top5_accuracy


# Register the model and adapter-aware loops in the existing command-line pipeline.
retrieval.EEGConformerVisualAdapterEncoder = EEGConformerVisualAdapterEncoder
retrieval.train_model = train_model_with_visual_adapter
retrieval.evaluate_model = evaluate_model_with_visual_adapter


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    retrieval.main()
