"""Frontend tests: mel exactness vs librosa + slice geometry (no weights needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mule_torch.config import MuleConfig
from mule_torch.frontend import MuleMelSpectrogram, slice_mel

librosa = pytest.importorskip("librosa")


def _librosa_mel(waveform: np.ndarray, cfg: MuleConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=cfg.window,
        center=cfg.center,
        pad_mode=cfg.pad_mode,
        power=cfg.power,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        norm=cfg.mel_norm,
        htk=cfg.htk,
    )
    return np.log10(10000.0 * mel + 1.0).astype(np.float32)


def test_mel_matches_librosa(waveform):
    cfg = MuleConfig()
    fe = MuleMelSpectrogram(cfg)  # builds the librosa filterbank internally
    with torch.no_grad():
        torch_mel = fe(torch.from_numpy(waveform).float().unsqueeze(0))[0].numpy()
    ref = _librosa_mel(waveform, cfg)
    k = min(torch_mel.shape[1], ref.shape[1])
    assert torch_mel.shape[0] == cfg.n_mels
    a, b = torch_mel[:, :k], ref[:, :k]
    # Max-abs is on the log10(10000*x+1) scale; ~1e-3 of float32 DFT-vs-FFT noise
    # near the mel floor is expected and harmless (cosine ~1, slices are renormed).
    assert np.abs(a - b).max() < 5e-3, np.abs(a - b).max()
    cos = float((a.reshape(-1) @ b.reshape(-1)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.99999, cos


def test_slice_geometry_and_normalization():
    cfg = MuleConfig()
    T = 1000  # frames
    mel = torch.randn(cfg.n_mels, T)
    slices = slice_mel(mel, cfg)
    # Centers at 200,400,... clamped into [150, T-150]; for T=1000 that's 200..800 -> 4 slices.
    expected_centers = [c for c in range(0, T, cfg.slice_hop) if c > 0]
    expected_centers = [min(max(c, cfg.look_backward), T - cfg.look_forward) for c in expected_centers]
    assert slices.shape == (len(expected_centers), 1, cfg.n_mels, cfg.slice_width)
    # Standard-normalized per slice: ~0 mean, ~1 std.
    m = slices.mean(dim=(1, 2, 3))
    s = slices.std(dim=(1, 2, 3), unbiased=False)
    assert torch.allclose(m, torch.zeros_like(m), atol=1e-5)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-3)


def test_short_clip_yields_one_slice():
    cfg = MuleConfig()
    mel = torch.randn(cfg.n_mels, 120)  # < slice_width (300)
    slices = slice_mel(mel, cfg)
    assert slices.shape == (1, 1, cfg.n_mels, cfg.slice_width)
