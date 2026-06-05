"""Unit tests for the primitive layers (no weights needed)."""

from __future__ import annotations

import numpy as np
import torch

from mule_torch.config import SCALED_GELU_GAIN, SCALED_RELU_GAIN, WS_EPS
from mule_torch.layers import SamePad2d, ScaledActivation, SqueezeExcite, WSConv2d, _same_pad_amounts


def test_same_pad_matches_tf_output_shape():
    # TF 'same' output length = ceil(in/stride); verify SamePad2d + valid conv gives that.
    for size, k, s in [(96, 3, 2), (300, 1, 8), (300, 7, 4), (24, 2, 2), (7, 2, 2)]:
        b0, b1 = _same_pad_amounts(size, k, s)
        out = (size + b0 + b1 - k) // s + 1
        assert out == -(-size // s)  # ceil division
        # TF pads more on the 'after' side when odd.
        assert b1 >= b0


def test_ws_standardization_matches_reference_formula():
    torch.manual_seed(0)
    conv = WSConv2d(8, 16, (3, 3), (1, 1), groups=1)
    conv.gain.data.uniform_(0.5, 1.5)
    w = conv.standardized_weight().detach().numpy()

    v = conv.weight.detach().numpy()  # (out, in/g, kH, kW)
    gain = conv.gain.detach().numpy()
    mean = v.mean(axis=(1, 2, 3), keepdims=True)
    var = v.var(axis=(1, 2, 3), keepdims=True)  # population
    fan_in = v[0].size
    scale = (1.0 / np.sqrt(np.maximum(var * fan_in, WS_EPS))) * gain.reshape(-1, 1, 1, 1)
    expected = (v - mean) * scale
    assert np.allclose(w, expected, atol=1e-6)
    # Standardized kernel is (near) zero-mean per output filter.
    assert np.allclose(w.mean(axis=(1, 2, 3)), 0.0, atol=1e-5)


def test_grouped_ws_conv_runs_and_shapes():
    conv = WSConv2d(768, 768, (1, 3), (2, 1), groups=6)
    x = torch.randn(2, 768, 12, 50)
    y = conv(x)
    assert y.shape[:2] == (2, 768)
    assert y.shape[2] == 6  # freq downsampled by stride 2 ('same': ceil(12/2))


def test_scaled_activations():
    x = torch.randn(4, 5)
    g = ScaledActivation("gelu")(x)
    assert torch.allclose(g, torch.nn.functional.gelu(x, approximate="none") * SCALED_GELU_GAIN)
    r = ScaledActivation("relu")(x)
    assert torch.allclose(r, torch.relu(x) * SCALED_RELU_GAIN)


def test_squeeze_excite_shape_and_range():
    se = SqueezeExcite(32)
    x = torch.randn(3, 32, 8, 10)
    y = se(x)
    assert y.shape == x.shape
