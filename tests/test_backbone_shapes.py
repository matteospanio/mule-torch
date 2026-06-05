"""Shape + structure tests for the backbone (no weights needed)."""

from __future__ import annotations

import torch

from mule_torch.backbone import SfNfNetF0
from mule_torch.blocks import NFNetBlock
from mule_torch.config import MuleConfig
from mule_torch.layers import SqueezeExcite, WSConv2d


def test_backbone_output_shape():
    model = SfNfNetF0().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 96, 300))
    assert out.shape == (2, 1728)


def test_intermediate_stage_shapes():
    cfg = MuleConfig()
    model = SfNfNetF0(cfg).eval()
    with torch.no_grad():
        _, st = model(torch.randn(1, 1, 96, 300), return_stages=True)
    # Slow path channels grow to the configured stage outputs.
    for i, c in enumerate(cfg.slow_stage_out):
        assert st[f"slow_stage{i}"].shape[1] == c
    for i, c in enumerate(cfg.fast_stage_out):
        assert st[f"fast_stage{i}"].shape[1] == c
    # The fused slow input to each stage has the REAL traced channel count.
    for i, c in enumerate(cfg.slow_stage_in):
        assert st[f"fuse{i}"].shape[1] == c
    assert st["embedding"].shape == (1, 1728)


def test_module_counts():
    model = SfNfNetF0()
    n_ws = sum(isinstance(m, WSConv2d) for m in model.modules())
    n_se = sum(isinstance(m, SqueezeExcite) for m in model.modules())
    n_blk = sum(isinstance(m, NFNetBlock) for m in model.modules())
    assert n_ws == 120, n_ws
    assert n_se == 24, n_se
    assert n_blk == 24, n_blk


def test_param_count_close_to_paper():
    model = SfNfNetF0()
    n = sum(p.numel() for p in model.parameters())
    # Paper states ~62.4M for SF-NFNet-F0.
    assert 55e6 < n < 70e6, f"{n/1e6:.2f}M params"
