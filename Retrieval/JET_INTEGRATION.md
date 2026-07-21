# JET EEG Retrieval Integration

This experiment adapts the architecture and conditional-flow-matching idea from
**Let EEG Models Learn EEG (JET, ICML 2026)** to the ATM EEG-to-image retrieval
pipeline.

## What is retained from JET

- raw EEG temporal patching;
- explicit preservation of channel identity;
- Transformer processing of EEG patch tokens;
- adaptive LayerNorm conditioning on continuous flow time;
- conditional flow matching as an auxiliary EEG representation objective.

## What is changed for this repository

The official JET model generates EEG. Here, its hidden representation is pooled
and projected into the 1024-D CLIP image space required by ATM. Retrieval is
trained with the original image/text alignment objective plus a weighted flow
matching loss.

## Run smoke tests

```bash
cd Retrieval
python test_jet_retrieval.py
```

## Train

```bash
cd Retrieval
python jet_retrieval.py \
  --encoder_type JETRetrievalEncoder \
  --data_path /path/to/Preprocessed_data_250Hz \
  --device cuda:0 \
  --epochs 40 \
  --batch_size 256
```

The default auxiliary flow weight is `0.1`. It is stored on the model as
`flow_weight` and should be ablated against `0.0`, `0.05`, `0.1`, and `0.2`.

## Required evaluation

Compare under identical subject splits and seeds:

1. original ATM encoder;
2. retrieval-specific EEG Conformer;
3. JET encoder with flow weight 0;
4. JET encoder with flow regularization.

Report 200-way Top-1 and Top-5, 2/4/10-way accuracy, validation alignment loss,
parameter count, and training time. Flow-generation results from the original
JET paper are not directly comparable to EEG-to-image retrieval results.

## References

- Y. Wang, Y. Ma, W. Li, and C. You, "Let EEG Models Learn EEG," ICML 2026.
- D. Li et al., "Visual Decoding and Reconstruction via EEG Embeddings with
  Guided Diffusion," NeurIPS 2024.
