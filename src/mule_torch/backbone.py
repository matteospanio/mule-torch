"""SF-NFNet-F0 backbone: log-mel slice ``(B,1,96,300)`` -> embedding ``(B,1728)``.

Assembles the slow/fast stems, fast-to-slow fusions, NFNet stages, and the
output head (global-average-pool both paths, concat ``[slow, fast]``, scaled
gelu) in the exact apply-order of ``sfnfnetf0.py:_apply_layers``.

This module contains NO frontend (mel / slicing): it operates on a single
standard-normalized log-mel slice, batched on the first axis. See
:mod:`mule_torch.frontend` and :mod:`mule_torch.model` for the full pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mule_torch.blocks import FastToSlowFusion, NFNetStage, Stem
from mule_torch.config import STAGE_DOWNSAMPLES, MuleConfig
from mule_torch.layers import ScaledActivation


class SfNfNetF0(nn.Module):
    """The MULE convolutional backbone (~62.4M params)."""

    def __init__(self, config: MuleConfig | None = None) -> None:
        super().__init__()
        cfg = config or MuleConfig()
        self.config = cfg
        bake = not cfg.weight_standardization
        act = cfg.scaled_activation

        self.slow_stem = Stem(cfg.slow_stem_channels, cfg.slow_stem_kernels, cfg.slow_stem_strides, act, 1, bake)
        self.fast_stem = Stem(cfg.fast_stem_channels, cfg.fast_stem_kernels, cfg.fast_stem_strides, act, 1, bake)

        betas = cfg.betas()
        # Fast channels entering each fusion = fast-stem out, then fast-stage outputs.
        fast_in_for_fusion = (cfg.fast_stem_channels[-1],) + cfg.fast_stage_out[:-1]

        self.fusions = nn.ModuleList(
            [
                FastToSlowFusion(
                    fast_in_channels=fi,
                    input_channels=ic,
                    output_channels=oc,
                    time_kernel=cfg.fusion_time_kernel,
                    time_stride=cfg.fusion_time_stride,
                    bake=bake,
                )
                for fi, ic, oc in zip(fast_in_for_fusion, cfg.fusion_input_channels, cfg.fusion_output_channels)
            ]
        )

        self.slow_stages = nn.ModuleList(
            [
                NFNetStage(
                    in_channels=sin,
                    out_channels=sout,
                    kernels=cfg.block_kernels,
                    freq_downsample=fds,
                    group_size=cfg.slow_group_size,
                    alpha=cfg.alpha,
                    betas=b,
                    scaled_activation=act,
                    bake=bake,
                )
                for sin, sout, fds, b in zip(cfg.slow_stage_in, cfg.slow_stage_out, STAGE_DOWNSAMPLES, betas)
            ]
        )
        self.fast_stages = nn.ModuleList(
            [
                NFNetStage(
                    in_channels=fin,
                    out_channels=fout,
                    kernels=cfg.block_kernels,
                    freq_downsample=fds,
                    group_size=cfg.fast_group_size,
                    alpha=cfg.alpha,
                    betas=b,
                    scaled_activation=act,
                    bake=bake,
                )
                for fin, fout, fds, b in zip(cfg.fast_stage_in, cfg.fast_stage_out, STAGE_DOWNSAMPLES, betas)
            ]
        )

        self.out_act = ScaledActivation(act)

    def forward(self, x: Tensor, return_stages: bool = False) -> Tensor | tuple[Tensor, dict]:
        """``(B,1,96,300)`` -> ``(B,1728)`` (or (emb, intermediates) if requested)."""
        stages: dict[str, Tensor] = {}
        slow = self.slow_stem(x)
        fast = self.fast_stem(x)
        if return_stages:
            stages["slow_stem"] = slow
            stages["fast_stem"] = fast

        for i, (fuse, slw, fst) in enumerate(zip(self.fusions, self.slow_stages, self.fast_stages)):
            slow = fuse(slow, fast)
            if return_stages:
                stages[f"fuse{i}"] = slow
            slow = slw(slow)
            fast = fst(fast)
            if return_stages:
                stages[f"slow_stage{i}"] = slow
                stages[f"fast_stage{i}"] = fast

        slow_pooled = slow.mean(dim=(2, 3))  # (B, 1536)
        fast_pooled = fast.mean(dim=(2, 3))  # (B, 192)
        emb = torch.cat([slow_pooled, fast_pooled], dim=1)  # (B, 1728)
        emb = self.out_act(emb)
        if return_stages:
            stages["embedding"] = emb
            return emb, stages
        return emb
