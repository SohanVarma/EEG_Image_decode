"""CPU smoke test for the retrieval-specific EEG Conformer encoder."""

import torch

from conformer_retrieval import EEGConformerRetrievalEncoder


def test_output_shape_and_normalization() -> None:
    model = EEGConformerRetrievalEncoder(
        input_shape=(63, 250),
        output_dim=1024,
        embed_dim=64,
        num_heads=4,
        depth=2,
        dropout=0.0,
    )
    model.eval()

    eeg = torch.randn(2, 63, 250)
    with torch.no_grad():
        embedding = model(eeg)

    assert embedding.shape == (2, 1024)
    norms = torch.linalg.vector_norm(embedding, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_factory_compatible_constructor() -> None:
    model = EEGConformerRetrievalEncoder((63, 250))
    assert model.input_shape == (63, 250)


if __name__ == "__main__":
    test_output_shape_and_normalization()
    test_factory_compatible_constructor()
    print("EEG Conformer retrieval smoke tests passed.")
