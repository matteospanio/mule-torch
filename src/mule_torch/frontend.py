"""Torch-native MULE frontend: waveform -> log-mel -> standard-normalized slices.

Two stages, both pure-torch (GPU-native, ONNX-friendly):

1. :class:`MuleMelSpectrogram` reproduces the upstream librosa mel exactly.
   The STFT is a fixed windowed-DFT ``conv1d`` (the "nnAudio" trick; see the
   ``ConvSTFT`` class — vendored from the parent project to stay self-contained)
   so it exports to ONNX, and the mel filterbank is the *librosa* filterbank
   (``norm=2.0``, which ``torchaudio`` cannot reproduce) stored as a buffer.
2. :func:`slice_mel` reproduces ``SliceExtractor``: 96x300 windows every 200
   frames (~2 s), each standard-normalized.

Config values come from ``mule_embedding_timeline.yml`` via :class:`MuleConfig`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mule_torch.config import MuleConfig


class ConvSTFT(nn.Module):
    """Power/magnitude spectrogram via a fixed windowed-DFT ``conv1d``.

    Vendored from the parent project's ``frontends/stft.py``. Numerically exact
    vs ``torch.stft`` / librosa for the same window + centering, and ONNX-safe
    (no complex tensors).
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
        window: str = "hann",
        center: bool = True,
        power: float = 2.0,
    ) -> None:
        super().__init__()
        if power not in (1.0, 2.0):
            raise ValueError(f"power must be 1.0 or 2.0, got {power}")
        win_length = win_length or n_fft
        if win_length > n_fft:
            raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.power = power
        self.pad = n_fft // 2

        if window == "hann":
            win = torch.hann_window(win_length, periodic=True)
        elif window == "none":
            win = torch.ones(win_length)
        else:
            raise ValueError(f"unsupported window {window!r}; use 'hann' or 'none'")

        if win_length < n_fft:  # center the short window inside the n_fft frame (librosa pad_center)
            padded = torch.zeros(n_fft)
            start = (n_fft - win_length) // 2
            padded[start : start + win_length] = win
            win = padded

        n_freqs = n_fft // 2 + 1
        k = torch.arange(n_freqs).unsqueeze(1)
        n = torch.arange(n_fft).unsqueeze(0)
        angle = 2.0 * math.pi * k * n / n_fft
        real = torch.cos(angle) * win
        imag = -torch.sin(angle) * win
        self.register_buffer("fwd_real", real.unsqueeze(1).float())
        self.register_buffer("fwd_imag", imag.unsqueeze(1).float())

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    def forward(self, waveform: Tensor) -> Tensor:
        x = waveform.unsqueeze(1)  # (B, 1, T)
        if self.center:
            x = F.pad(x, (self.pad, self.pad), mode="reflect")
        real = F.conv1d(x, self.fwd_real, stride=self.hop_length)
        imag = F.conv1d(x, self.fwd_imag, stride=self.hop_length)
        spec = real * real + imag * imag
        if self.power == 1.0:
            spec = torch.sqrt(spec + 1e-12)
        return spec


def build_mel_filterbank(cfg: MuleConfig) -> Tensor:
    """Build the librosa mel filterbank ``(n_mels, n_freqs)`` for ``cfg``.

    Requires librosa (the ``frontend`` extra). The filterbank is normally
    precomputed at conversion time and stored in the safetensors, so inference
    does not call this.
    """
    import librosa  # local import: only needed at build/convert time

    fb = librosa.filters.mel(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        htk=cfg.htk,
        norm=cfg.mel_norm,
    )  # (n_mels, n_freqs)
    return torch.from_numpy(fb).float()


class MuleMelSpectrogram(nn.Module):
    """``(B, T)`` waveform @16k -> ``(B, n_mels, frames)`` MULE log-mel.

    The mel filterbank is a buffer; pass it in (from converted weights) or let
    it build via librosa. Compression is ``log10(10000*x + 1)`` (``log10_nonneg``)
    and ``mag_range`` is null (no max-subtraction).
    """

    def __init__(self, cfg: MuleConfig, mel_fb: Tensor | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.stft = ConvSTFT(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=cfg.window,
            center=cfg.center,
            power=cfg.power,
        )
        if mel_fb is None:
            mel_fb = build_mel_filterbank(cfg)
        if mel_fb.shape != (cfg.n_mels, cfg.n_freqs):
            raise ValueError(f"mel_fb shape {tuple(mel_fb.shape)} != {(cfg.n_mels, cfg.n_freqs)}")
        self.register_buffer("mel_fb", mel_fb.float())

    def forward(self, waveform: Tensor) -> Tensor:
        spec = self.stft(waveform)              # (B, n_freqs, frames) power
        mel = torch.matmul(self.mel_fb, spec)   # (B, n_mels, frames)
        if self.cfg.mag_compression == "log10_nonneg":
            mel = torch.log10(10000.0 * mel + 1.0)
        elif self.cfg.mag_compression in ("linear", None):
            pass
        else:
            raise ValueError(f"unsupported mag_compression {self.cfg.mag_compression!r}")
        return mel


def slice_mel(mel: Tensor, cfg: MuleConfig) -> Tensor:
    """Reproduce ``SliceExtractor`` for a single clip's mel ``(n_mels, T)``.

    Returns standard-normalized slices ``(N, 1, n_mels, slice_width)``.
    """
    n_mels, T = mel.shape
    lb, lf, hop, w = cfg.look_backward, cfg.look_forward, cfg.slice_hop, cfg.slice_width

    if T < w:
        # Too short for a real window: pad time up to one full slice. Reflect is
        # limited to pad < T, so use replicate (edge repeat) which accepts any pad.
        pad = w - T
        mode = "reflect" if pad < T else "replicate"
        mel = F.pad(mel.unsqueeze(0), (0, pad), mode=mode).squeeze(0)
        centers = [lb]
        T = w
    else:
        centers = [t for t in range(0, T, hop) if t > 0]
        centers = [min(max(c, lb), T - lf) for c in centers]
        if not centers:  # 200 < T < 300 corner: take the last valid window
            centers = [T - lf]

    slices = torch.stack([mel[:, c - lb : c + lf] for c in centers], dim=0)  # (N, n_mels, w)
    slices = slices.unsqueeze(1)  # (N, 1, n_mels, w)

    if cfg.standard_normalize:
        mean = slices.mean(dim=(1, 2, 3), keepdim=True)
        slices = slices - mean
        std = slices.std(dim=(1, 2, 3), unbiased=False, keepdim=True)
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std = torch.clamp(std, min=cfg.std_floor)
        slices = slices / std
    return slices
