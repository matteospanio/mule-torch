# mule-torch

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Weights: CC BY-NC 4.0](https://img.shields.io/badge/Weights-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE.weights)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.1-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-unofficial%20port-orange.svg)](#disclaimer)

An **unofficial PyTorch port** of MULE (Musicset Unsupervised Large Embedding), the
SF-NFNet-F0 music-audio representation model from SiriusXM/Pandora:

> *Supervised and Unsupervised Learning of Audio Representations for Music
> Understanding*, M. C. McCallum, F. Korzeniowski, S. Oramas, F. Gouyon,
> A. F. Ehmann. ISMIR 2022. <https://arxiv.org/abs/2210.03799>

This is **not a re-training**. It re-implements the SF-NFNet-F0 architecture in
pure PyTorch and **transfers the pretrained weights** from the original Keras
model (`model.keras`), verified to be numerically equivalent. The result is a
clean `nn.Module` that is **batched, GPU-native, and ONNX-exportable** — none of
which the original TensorFlow/SCOOCH/`Analysis` pipeline supports.

> ### Disclaimer
> This is an **independent, community port from TensorFlow to PyTorch**. It is
> **not affiliated with, endorsed by, or maintained by** SiriusXM, Pandora, or
> the original authors. The original model, weights, and configurations come
> from [PandoraMedia/music-audio-representations](https://github.com/PandoraMedia/music-audio-representations).
> All credit for the model goes to the original authors — please cite their paper.

## Install

```bash
pip install mule-torch                                  # once published to PyPI
# or, from source:
pip install git+https://github.com/matteospanio/mule-torch
```

The installed package is a **pure torch library** (`torch`, `numpy`,
`safetensors`, `huggingface_hub`). It does **not** pull in TensorFlow — the
conversion/verification tooling lives in standalone `uv` scripts (see below).

## Usage

```python
from mule_torch import MuleModel

model = MuleModel.from_pretrained(model_dir="artifacts")   # or hf_repo="..."
emb = model(waveform)        # waveform: (B, T) float @ 16 kHz mono -> (B, 1728)
```

Input is a 16 kHz mono waveform in `[-1, 1]`. The model computes a 96-band
log-mel spectrogram, slices it into 96×300 windows every ~2 s, runs the
SF-NFNet-F0 backbone, and mean-pools the per-slice 1728-d embeddings into one
vector per clip — matching `mule_embedding_timeline.yml` + a timeline average.

## How the port works

| Stage | Original (TF) | This port (torch) |
|---|---|---|
| Mel front-end | `librosa.feature.melspectrogram` | `MuleMelSpectrogram` (fixed windowed-DFT `conv1d` + the **librosa** filterbank stored as a buffer, since `torchaudio` can't do `norm=2.0`) |
| Slicing | `SliceExtractor` (numpy) | `slice_mel` (torch) |
| Backbone | `SfNfNetF0` Keras model | `SfNfNetF0` `nn.Module` (`WSConv2d`, scaled GELU, squeeze-excite, NFNet blocks, fast→slow fusion) |
| Weights | `model.keras` (251 MB) | `model.safetensors` (converted) |

Weight standardization is recomputed on the fly (faithful + fine-tunable);
constants (β, α, scaled-activation gains) are baked into the architecture; the
learnable skip-init gains are the only saved scalars per block. Stochastic depth
is a no-op at inference (`shortcut + residual`) and is dropped.

> **Amplitude convention.** The original `AudioFile` reader scales PCM16 by
> `1/2^16`. If you feed conventional `[-1,1]` audio, embeddings still track the
> original closely but are not bit-identical because the `log10(10000·x+1)` mel
> compression is non-linear. The verification below feeds the *exact* waveform
> the reference used, so parity is exact.

## Layout

```
src/mule_torch/   config.py layers.py blocks.py backbone.py frontend.py model.py   # the library
scripts/          convert.py  verify.py   # standalone uv tools (NOT part of the package)
tests/            frontend / shapes / layers / onnx (no weights)  + parity (gated)
references/       vendored upstream TF repo + paper (gitignored; conversion only)
```

## Converting + verifying the weights

The conversion (TF → safetensors) and the parity check are **standalone
[`uv`](https://docs.astral.sh/uv/) scripts** with [PEP 723](https://peps.python.org/pep-0723/)
inline dependencies — no virtualenv setup, no TensorFlow in the package. Just
`uv run` them; uv builds the right ephemeral environment (Python ≤ 3.11 for TF).

```bash
# 0) get the 251 MB Keras weights + reference code
git clone https://github.com/PandoraMedia/music-audio-representations.git references/music-audio-representations
( cd references/music-audio-representations && git lfs pull )
REF=references/music-audio-representations

# 1a) EXTRACT: model.keras -> weights.npz  (TensorFlow)
uv run scripts/convert.py extract \
    --keras $REF/supporting_data/model/model.keras --references $REF --out artifacts/weights.npz

# 1b) ASSEMBLE: weights.npz -> model.safetensors + config.json  (torch)
uv run scripts/convert.py assemble --npz artifacts/weights.npz --out artifacts

# 2) Parity: genuine TF pipeline vs the torch port, end-to-end + ONNX
uv run scripts/verify.py reference --references $REF \
    --config $REF/supporting_data/configs/mule_embedding_timeline.yml \
    --wav tests/fixtures/fixture.wav --out artifacts/ref
uv run scripts/verify.py compare --ref artifacts/ref --weights artifacts --onnx
```

### Tests

```bash
uv pip install -e ".[dev]"
pytest -m "not requires_weights"                                  # runs anywhere
MULE_TORCH_WEIGHTS=artifacts MULE_TF_REF=artifacts/ref pytest      # incl. gated parity
```

Tests that need no weights (frontend exactness vs librosa, traced shapes, layer
math, ONNX-backbone parity) run anywhere; the parity tests are skipped unless the
two env vars above point at converted weights + reference dumps.

## Verified parity

On an RTX 3070 against the genuine TF MULE pipeline:

- Mel vs librosa: cosine `1.0000` (max-abs drift washes out after per-slice norm).
- Backbone on reference slices: cosine `1.0000000`.
- **End-to-end clip embedding vs original MULE: cosine `0.9999999`.**
- ONNX backbone vs torch: max-abs `< 1e-6`.
- Parameter count: `62.35M` (paper: ~62.4M).

## Licensing

- **Code:** GPL-3.0-only (mirrors the upstream `mule` module). See [`LICENSE`](LICENSE).
- **Converted weights:** CC BY-NC 4.0 (inherited from the upstream MULE weights —
  non-commercial). See [`LICENSE.weights`](LICENSE.weights).

Please cite McCallum et al. (2022) if you use this. See [`NOTICE`](NOTICE) for provenance.
