"""CPU smoke tests for the residual CLIP visual adapter."""

import torch

from clip_visual_adapter_retrieval import (
    EEGConformerVisualAdapterEncoder,
    ResidualVisualAdapter,
)


def test_adapter_starts_as_identity_direction() -> None:
    adapter = ResidualVisualAdapter(
        feature_dim=16,
        bottleneck_dim=4,
        dropout=0.0,
    )
    adapter.eval()

    features = torch.randn(3, 16)
    with torch.no_grad():
        adapted = adapter(features)
        expected = torch.nn.functional.normalize(features, dim=-1)

    assert adapted.shape == features.shape
    assert torch.allclose(adapted, expected, atol=1e-6)


def test_adapter_receives_gradients() -> None:
    adapter = ResidualVisualAdapter(
        feature_dim=16,
        bottleneck_dim=4,
        dropout=0.0,
    )
    features = torch.randn(3, 16)
    target = torch.randn(3, 16)

    loss = 1.0 - torch.nn.functional.cosine_similarity(
        adapter(features), target, dim=-1
    ).mean()
    loss.backward()

    assert adapter.up.weight.grad is not None
    assert torch.isfinite(adapter.up.weight.grad).all()


def test_combined_model_shapes() -> None:
    model = EEGConformerVisualAdapterEncoder(
        input_shape=(63, 250),
        output_dim=1024,
        embed_dim=64,
        num_heads=4,
        depth=2,
        dropout=0.0,
        adapter_bottleneck=128,
        adapter_dropout=0.0,
    )
    model.eval()

    eeg = torch.randn(2, 63, 250)
    image_features = torch.randn(2, 1024)
    with torch.no_grad():
        eeg_embedding = model(eeg)
        image_embedding = model.adapt_image_features(image_features)

    assert eeg_embedding.shape == (2, 1024)
    assert image_embedding.shape == (2, 1024)
    assert torch.allclose(
        torch.linalg.vector_norm(eeg_embedding, dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(image_embedding, dim=-1),
        torch.ones(2),
        atol=1e-5,
    )


if __name__ == "__main__":
    test_adapter_starts_as_identity_direction()
    test_adapter_receives_gradients()
    test_combined_model_shapes()
    print("CLIP visual adapter smoke tests passed.")
