"""CPU smoke tests for the JET-style EEG retrieval encoder."""

import torch

from jet_retrieval import JETRetrievalEncoder


def test_output_shape_and_normalization() -> None:
    model = JETRetrievalEncoder(
        input_shape=(63, 250),
        output_dim=1024,
        patch_size=25,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        dropout=0.0,
    )
    model.eval()
    eeg = torch.randn(2, 63, 250)
    with torch.no_grad():
        embedding = model(eeg)

    assert embedding.shape == (2, 1024)
    norms = torch.linalg.vector_norm(embedding, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_flow_loss_is_finite_and_differentiable() -> None:
    model = JETRetrievalEncoder(
        input_shape=(63, 250),
        output_dim=128,
        patch_size=25,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        dropout=0.0,
    )
    eeg = torch.randn(2, 63, 250)
    loss = model.flow_matching_loss(eeg)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.velocity_head[-1].weight.grad is not None


def test_factory_constructor() -> None:
    model = JETRetrievalEncoder((63, 250))
    assert model.input_shape == (63, 250)
    assert model.output_dim == 1024


if __name__ == "__main__":
    test_output_shape_and_normalization()
    test_flow_loss_is_finite_and_differentiable()
    test_factory_constructor()
    print("JET EEG retrieval smoke tests passed.")
