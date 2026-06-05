"""Single source of truth for every MULE architecture / frontend / slicing constant.

All numbers here are read off the upstream TensorFlow implementation
(``mule/models/sfnfnetf0.py`` and ``supporting_data/configs/mule_embedding_timeline.yml``)
so that the PyTorch port is numerically equivalent. Nothing here is a "nice
round default" — changing a value will break parity with the released weights.

The :class:`MuleConfig` is serialised verbatim into ``config.json`` next to the
converted ``model.safetensors``, and rebuilt by :meth:`MuleModel.from_pretrained`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# --- Scaled (variance-preserving) activation gains, from sfnfnetf0.py -------
# These exact float32 constants come from the DeepMind NFNet reference code and
# are baked into the released weights; do not "tidy" them.
SCALED_GELU_GAIN = 1.7015043497085571
SCALED_RELU_GAIN = 1.7139588594436646

# --- Weight-standardization epsilon (WeightStandardization._compute_weights) -
WS_EPS = 1e-4

# --- NFNet residual scaling --------------------------------------------------
ALPHA = 0.2  # constant residual-tail multiplier on every block
NFNET_STAGE_DEPTHS = (1, 2, 6, 3)  # F0: blocks per stage (F-value 0)
STAGE_DOWNSAMPLES = (1, 2, 2, 2)  # frequency downsample factor at each stage's first block


def stage_expected_vars(alpha: float = ALPHA) -> list[float]:
    """Expected input variance at each stage (sfnfnetf0.py ``_stage_expected_vars``)."""
    return [1.0] + [(1.0 + alpha**2) ** 0.5] * 3


def block_betas(alpha: float = ALPHA) -> list[list[float]]:
    """Per-block beta constants, one list per stage.

    Mirrors ``_make_nfnet_stage``: the first block of a stage uses
    ``beta = 1 / input_expected_var``; subsequent blocks track a running
    ``expected_std`` updated as ``sqrt(expected_std**2 + alpha**2)`` and use
    ``beta = 1 / expected_std``.
    """
    betas: list[list[float]] = []
    for depth, exp_var in zip(NFNET_STAGE_DEPTHS, stage_expected_vars(alpha)):
        stage = [1.0 / exp_var]
        expected_std = (exp_var**2 + alpha**2) ** 0.5
        for _ in range(1, depth):
            stage.append(1.0 / expected_std)
            expected_std = (expected_std**2 + alpha**2) ** 0.5
        betas.append(stage)
    return betas


@dataclass
class MuleConfig:
    """All constants needed to rebuild the MULE PyTorch model + frontend."""

    # ---- output ----
    embedding_dim: int = 1728  # 1536 (slow GAP) + 192 (fast GAP)

    # ---- backbone: stems (channels / kernels / strides per conv) ----
    slow_stem_channels: tuple[int, ...] = (16, 32, 64, 128)
    slow_stem_kernels: tuple[tuple[int, int], ...] = ((3, 1), (3, 1), (3, 1), (3, 3))
    slow_stem_strides: tuple[tuple[int, int], ...] = ((2, 8), (1, 1), (1, 1), (2, 2))
    fast_stem_channels: tuple[int, ...] = (2, 4, 8, 16)
    fast_stem_kernels: tuple[tuple[int, int], ...] = ((3, 3), (3, 3), (3, 3), (3, 3))
    fast_stem_strides: tuple[tuple[int, int], ...] = ((2, 2), (1, 1), (1, 1), (2, 2))

    # ---- backbone: NFNet stages (REAL traced channel counts) ----
    # Slow stage-3 input is 4608 (concat of slow 1536 + fusion3 expansion 3072),
    # NOT the advisory 2560 in the upstream source. See plan / sfnfnetf0.py comment.
    slow_stage_in: tuple[int, ...] = (256, 512, 1024, 4608)
    slow_stage_out: tuple[int, ...] = (256, 512, 1536, 1536)
    fast_stage_in: tuple[int, ...] = (16, 32, 64, 192)
    fast_stage_out: tuple[int, ...] = (32, 64, 192, 192)
    slow_group_size: int = 128
    fast_group_size: int = 16
    block_kernels: tuple[tuple[int, int], ...] = ((1, 1), (1, 3), (3, 1), (1, 1))

    # ---- backbone: fast-to-slow fusion (conv1 out = "input_channels",
    #      conv2 out = "output_channels" expansion concatenated into slow) ----
    fusion_input_channels: tuple[int, ...] = (32, 32, 64, 192)
    fusion_output_channels: tuple[int, ...] = (128, 256, 512, 3072)
    fusion_time_kernel: int = 7
    fusion_time_stride: int = 4

    alpha: float = ALPHA
    scaled_activation: str = "gelu"
    ws_eps: float = WS_EPS
    weight_standardization: bool = True  # False => kernels pre-baked (plain conv)

    # ---- frontend: mel spectrogram (mule_embedding_timeline.yml) ----
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 160
    win_length: int = 400
    window: str = "hann"
    n_mels: int = 96
    fmin: float = 0.0
    fmax: float = 8000.0
    mel_norm: float = 2.0
    htk: bool = True
    power: float = 2.0
    center: bool = True
    pad_mode: str = "reflect"
    mag_compression: str = "log10_nonneg"  # log10(10000*x + 1)

    # ---- slicing (SliceExtractor) ----
    slice_hop: int = 200
    look_backward: int = 150
    look_forward: int = 150
    standard_normalize: bool = True
    std_floor: float = 0.01

    # ---- provenance ----
    provenance: dict[str, Any] = field(
        default_factory=lambda: {
            "source_repo": "https://github.com/PandoraMedia/music-audio-representations",
            "paper": "https://arxiv.org/abs/2210.03799",
            "model": "SF-NFNet-F0 (MULE)",
            "config": "supporting_data/configs/mule_embedding_timeline.yml",
            "code_license": "GPL-3.0-only",
            "weights_license": "CC-BY-NC-4.0",
        }
    )

    # ---- derived ----
    @property
    def slice_width(self) -> int:
        """Time width of one slice fed to the backbone (look_back + look_fwd)."""
        return self.look_backward + self.look_forward

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    def betas(self) -> list[list[float]]:
        return block_betas(self.alpha)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MuleConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        # Coerce JSON lists back into tuples for the tuple-typed fields.
        tuple_fields = {
            name
            for name, fld in cls.__dataclass_fields__.items()  # type: ignore[attr-defined]
            if "tuple" in str(fld.type)
        }
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k not in known:
                continue
            if k in tuple_fields and isinstance(v, list):
                v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
            out[k] = v
        return cls(**out)


# Sanity: the slow/fast GAP outputs must sum to embedding_dim.
assert MuleConfig().slow_stage_out[-1] + MuleConfig().fast_stage_out[-1] == MuleConfig().embedding_dim
# Sanity: slice width matches the upstream 300-frame (3 s @ hop 160) window.
assert MuleConfig().slice_width == 300
# Sanity: beta tables have the right shape.
assert [len(s) for s in MuleConfig().betas()] == list(NFNET_STAGE_DEPTHS)
