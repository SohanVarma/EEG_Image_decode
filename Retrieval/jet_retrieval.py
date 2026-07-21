"""Train ATM retrieval with the modular JET-style EEG encoder.

Run from Retrieval:

    python jet_retrieval.py \
        --encoder_type JETRetrievalEncoder \
        --data_path /path/to/Preprocessed_data_250Hz \
        --device cuda:0 --epochs 40 --batch_size 256
"""

from __future__ import annotations

import builtins

import torch
from torch import Tensor

# The upstream ATM script uses Tensor annotations without importing Tensor.
builtins.Tensor = Tensor

import contrast_retrieval as retrieval
from jet_encoder import JETRetrievalEncoder


# Register the encoder with the existing ATM model factory.
retrieval.JETRetrievalEncoder = JETRetrievalEncoder
_original_train_model = retrieval.train_model


def train_model_with_jet_flow(
    model,
    dataloader,
    optimizer,
    device,
    text_features_all,
    img_features_all,
):
    """Add flow regularization only when training the JET encoder."""
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
