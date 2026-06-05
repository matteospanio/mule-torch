"""Primitive layers for the MULE SF-NFNet-F0 port.

Each layer mirrors an upstream TensorFlow construct exactly:

- :class:`SamePad2d`       <- TF ``padding='same'`` (asymmetric)
- :class:`WSConv2d`        <- ``WeightStandardization(Conv2D(...))``
- :class:`ScaledActivation`<- ``get_scaled_activation('gelu'|'relu')``
- :class:`SqueezeExcite`   <- ``_make/_apply_squeeze_and_excite``

References: ``mule/models/layers/{weight_standardization,scalar_multiply}.py``,
``mule/models/sfnfnetf0.py``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mule_torch.config import SCALED_GELU_GAIN, SCALED_RELU_GAIN, WS_EPS


def _same_pad_amounts(size: int, kernel: int, stride: int) -> tuple[int, int]:
    """TF ``SAME`` padding for one dimension (dilation 1).

    Returns (pad_before, pad_after). TF pads MORE on the after (right/bottom)
    side when the total padding is odd.
    """
    out = math.ceil(size / stride)
    pad_total = max((out - 1) * stride + kernel - size, 0)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return pad_before, pad_after


class SamePad2d(nn.Module):
    """Apply TF-style ``SAME`` zero-padding before a stride-valid conv/pool.

    Width is the *time* axis (W), height the *frequency* axis (H). The pad is
    data-shape dependent, so it is computed in ``forward`` (cheap, integer).
    """

    def __init__(self, kernel: tuple[int, int], stride: tuple[int, int]) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        h, w = x.shape[-2], x.shape[-1]
        ph0, ph1 = _same_pad_amounts(h, self.kernel[0], self.stride[0])
        pw0, pw1 = _same_pad_amounts(w, self.kernel[1], self.stride[1])
        if ph0 or ph1 or pw0 or pw1:
            # F.pad order: (W_before, W_after, H_before, H_after)
            x = F.pad(x, (pw0, pw1, ph0, ph1))
        return x


class ScaledActivation(nn.Module):
    """Variance-preserving GELU/ReLU: ``act(x) * gain`` (sfnfnetf0.py)."""

    def __init__(self, kind: str = "gelu") -> None:
        super().__init__()
        if kind not in ("gelu", "relu"):
            raise ValueError(f"scaled activation must be 'gelu' or 'relu', got {kind!r}")
        self.kind = kind
        self.gain = SCALED_GELU_GAIN if kind == "gelu" else SCALED_RELU_GAIN

    def forward(self, x: Tensor) -> Tensor:
        if self.kind == "gelu":
            # TF tf.nn.gelu default is exact erf-gelu (approximate=False).
            return F.gelu(x, approximate="none") * self.gain
        return F.relu(x) * self.gain


class WSConv2d(nn.Module):
    """Weight-standardized 2D convolution (no bias), matching the TF wrapper.

    Stores the raw kernel ``weight`` shaped ``(out, in/groups, kH, kW)``, a
    per-output-channel ``gain``, and a ``bias`` (the wrapped ``Conv2D`` in the
    upstream code keeps Keras' default ``use_bias=True``; the bias is added
    *after* the standardized convolution). At ``forward`` the standardized
    kernel is recomputed exactly as ``WeightStandardization._compute_weights``::

        mean = reduce_mean(v, axis=(0,1,2))           # over (kH,kW,in/g)
        var  = reduce_variance(v, axis=(0,1,2))       # population variance
        fan_in = prod(kernel.shape[:-1])              # (in/g)*kH*kW
        scale = rsqrt(max(var*fan_in, eps)) * gain
        kernel = v*scale - mean*scale = (v - mean)*scale

    In torch the reduction axes (kH,kW,in/g) map to dims (1,2,3) of the
    ``(out,in/g,kH,kW)`` layout. ``bake_=True`` freezes the standardized kernel
    into a plain buffer (inference-only, smaller ONNX) instead of recomputing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        groups: int = 1,
        eps: float = WS_EPS,
        bake: bool = False,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError(f"in_channels {in_channels} not divisible by groups {groups}")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels {out_channels} not divisible by groups {groups}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.groups = groups
        self.eps = eps
        self.bake = bake
        self.pad = SamePad2d(kernel, stride)

        kh, kw = kernel
        weight = torch.empty(out_channels, in_channels // groups, kh, kw)
        nn.init.kaiming_normal_(weight)
        self.weight = nn.Parameter(weight)
        # Wrapped Conv2D keeps Keras' default use_bias=True (init zeros).
        self.bias = nn.Parameter(torch.zeros(out_channels))
        if bake:
            # Inference-only: the standardized kernel is the parameter.
            self.register_parameter("gain", None)
        else:
            self.gain = nn.Parameter(torch.ones(out_channels))

    def standardized_weight(self) -> Tensor:
        if self.bake:
            return self.weight
        v = self.weight
        mean = v.mean(dim=(1, 2, 3), keepdim=True)
        var = v.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        fan_in = v[0].numel()  # (in/g)*kH*kW == prod(TF kernel.shape[:-1])
        scale = torch.rsqrt(torch.clamp(var * fan_in, min=self.eps)) * self.gain.view(-1, 1, 1, 1)
        return (v - mean) * scale

    def forward(self, x: Tensor) -> Tensor:
        w = self.standardized_weight()
        x = self.pad(x)
        return F.conv2d(x, w, bias=self.bias, stride=self.stride, padding=0, groups=self.groups)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation block matching ``_apply_squeeze_and_excite``.

    GAP -> Dense(C//2, relu) -> Dense(C, sigmoid) -> (*2.0) -> broadcast-multiply.
    Plain (unscaled) relu/sigmoid, with bias, exactly as upstream.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 2, bias=True)
        self.fc2 = nn.Linear(channels // 2, channels, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W); global average over spatial dims.
        s = x.mean(dim=(2, 3))           # (B, C)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))   # (B, C)
        s = (s * 2.0).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * s
