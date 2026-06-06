"""The batched backbone path must equal the old per-clip path (no weights).

``embed_timeline`` now concatenates every clip's slices into one backbone call
instead of looping per clip. Because SF-NFNet-F0 is normalizer-free, that is
supposed to be numerically identical. This guards that invariant on a
random-init model so it runs anywhere (no converted weights required).
"""

from __future__ import annotations

import torch

from mule_torch import MuleModel
from mule_torch.config import MuleConfig
from mule_torch.frontend import slice_mel


def _reference_timeline(model: MuleModel, waveform: torch.Tensor) -> list[torch.Tensor]:
    """The pre-refactor algorithm: one backbone call per clip."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mels = model.frontend(waveform)
    return [model.backbone(slice_mel(mels[b], model.config)) for b in range(mels.shape[0])]


def _model() -> MuleModel:
    torch.manual_seed(0)
    return MuleModel(MuleConfig()).eval()


def test_batched_timeline_matches_per_clip():
    model = _model()
    sr = model.sample_rate
    # A 3-clip batch, each long enough for several ~2 s slices.
    torch.manual_seed(1)
    waveform = torch.randn(3, sr * 8) * 0.1
    with torch.no_grad():
        got = model.embed_timeline(waveform)
        ref = _reference_timeline(model, waveform)
    assert len(got) == len(ref) == 3
    for g, r in zip(got, ref):
        assert g.shape == r.shape
        assert torch.allclose(g, r, atol=1e-5, rtol=1e-4), (g - r).abs().max().item()


def test_forward_shape_and_pooling():
    model = _model()
    sr = model.sample_rate
    waveform = torch.randn(2, sr * 8) * 0.1
    with torch.no_grad():
        emb = model(waveform)
        timeline = model.embed_timeline(waveform)
    assert emb.shape == (2, 1728)
    # forward() is just the per-clip mean over the timeline.
    expected = torch.stack([t.mean(dim=0) for t in timeline], dim=0)
    assert torch.allclose(emb, expected, atol=1e-6)


def test_single_clip_1d_input():
    model = _model()
    sr = model.sample_rate
    with torch.no_grad():
        emb = model(torch.randn(sr * 8) * 0.1)
    assert emb.shape == (1, 1728)
