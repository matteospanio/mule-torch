"""Shared fixtures + weight/reference gating."""

from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(scope="session")
def waveform() -> np.ndarray:
    """A deterministic ~6 s, 16 kHz mono waveform in [-1, 1]."""
    sr, seconds = 16000, 6.0
    n = int(sr * seconds)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    x = (
        0.5 * np.sin(2 * np.pi * (220.0 + 40.0 * t) * t)
        + 0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2 * np.pi * 880.0 * t)
        + 0.02 * rng.standard_normal(n)
    ).astype(np.float32)
    return x / (np.max(np.abs(x)) + 1e-6)


@pytest.fixture(scope="session")
def weights_dir() -> str:
    d = os.environ.get("MULE_TORCH_WEIGHTS")
    if not d or not os.path.exists(os.path.join(d, "model.safetensors")):
        pytest.skip("MULE_TORCH_WEIGHTS not set / no converted weights present")
    return d


@pytest.fixture(scope="session")
def tf_ref_dir() -> str:
    d = os.environ.get("MULE_TF_REF")
    if not d or not os.path.exists(os.path.join(d, "timeline.npy")):
        pytest.skip("MULE_TF_REF not set / no reference dumps present")
    return d
