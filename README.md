# mule-torch

A faithful **PyTorch port of MULE** (Musicset Unsupervised Large Embedding), the
SF-NFNet-F0 music-audio representation model from SiriusXM/Pandora:

> *Supervised and Unsupervised Learning of Audio Representations for Music
> Understanding*, M. C. McCallum, F. Korzeniowski, S. Oramas, F. Gouyon,
> A. F. Ehmann. ISMIR 2022. <https://arxiv.org/abs/2210.03799>

This is **not a re-training**. It re-implements the SF-NFNet-F0 architecture in
pure PyTorch and **transfers the pretrained weights** from the released Keras
model (`model.keras`), verified to be numerically equivalent. The result is a
clean `nn.Module` that is **batched, GPU-native, and ONNX-exportable** — none of
which the original TensorFlow/SCOOCH/`Analysis` pipeline supports.

```python
from mule_torch import MuleModel

model = MuleModel.from_pretrained(model_dir="artifacts")   # or hf_repo=...
emb = model(waveform)        # waveform: (B, T) float @ 16 kHz mono -> (B, 1728)
```

Input is a 16 kHz mono waveform in `[-1, 1]`. The model computes a 96-band
log-mel spectrogram, slices it into 96×300 windows every ~2 s, runs the
SF-NFNet-F0 backbone, and mean-pools the per-slice 1728-d embeddings into one
vector per clip — exactly matching `mule_embedding_timeline.yml` + a timeline
average.

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
src/mule_torch/   config.py layers.py blocks.py backbone.py frontend.py model.py
scripts/          convert.py (TF -> safetensors)   verify.py (parity + ONNX)
tests/            frontend / shapes / layers / onnx (no weights)  + parity (gated)
references/       vendored upstream TF repo + paper (gitignored; conversion only)
```

## Reproduce the conversion + verification

Two environments are needed because TensorFlow 2.13 requires Python ≤ 3.11 and is
*not* a runtime dependency of this package.

```bash
# 0) get the 251 MB Keras weights
cd references/music-audio-representations && git lfs pull && cd -

# 1) conversion + reference dumps  (TF venv, Python 3.10)
uv venv --python 3.10 .venv-convert
uv pip install --python .venv-convert -e ".[convert]"
.venv-convert/bin/python scripts/convert.py \
    --keras references/music-audio-representations/supporting_data/model/model.keras \
    --references references/music-audio-representations --out artifacts
.venv-convert/bin/python scripts/verify.py reference \
    --references references/music-audio-representations \
    --config references/music-audio-representations/supporting_data/configs/mule_embedding_timeline.yml \
    --wav tests/fixtures/fixture.wav --out artifacts/ref

# 2) torch-side parity + ONNX  (runtime venv, Python 3.12 + CUDA)
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
.venv/bin/python scripts/verify.py compare --ref artifacts/ref --weights artifacts --onnx

# 3) full test suite (parity tests need the env vars below)
MULE_TORCH_WEIGHTS=artifacts MULE_TF_REF=artifacts/ref .venv/bin/python -m pytest
```

Tests that need no weights (frontend exactness vs librosa, traced shapes, layer
math, ONNX-backbone parity) run anywhere: `pytest -m "not requires_weights"`.

## Acceptance

- Mel vs librosa: max-abs < 1e-3.
- Backbone on reference slices: cosine ≥ 0.9999.
- End-to-end clip embedding vs original MULE: cosine ≥ 0.9999.
- ONNX backbone vs torch: max-abs < 1e-4.

## Licensing

- **Code:** GPL-3.0-only (mirrors the upstream `mule` module). See `LICENSE`.
- **Converted weights:** CC BY-NC 4.0 (inherited from the upstream MULE weights —
  non-commercial). See `LICENSE.weights`.

Please cite McCallum et al. (2022) if you use this. See `NOTICE` for provenance.
