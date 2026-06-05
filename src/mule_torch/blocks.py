"""Composite blocks for SF-NFNet-F0: Stem, NFNetBlock, NFNetStage, FastToSlowFusion.

Mirrors ``mule/models/sfnfnetf0.py``:
``_make_stem_module`` / ``_make_nfnet_block`` / ``_make_nfnet_stage`` /
``_make_fast_to_slow_fusion`` and their ``_apply_*`` counterparts.

Stochastic depth is dropped: at inference (``scale_during_test=False``) it is
exactly ``shortcut + residual``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mule_torch.layers import SamePad2d, ScaledActivation, SqueezeExcite, WSConv2d


class TfSameAvgPool2d(nn.Module):
    """Average pooling with TF ``padding='same'`` semantics.

    TF averages over only the *valid* (non-padded) elements in each window
    (``count_include_pad=False``). We zero-pad asymmetrically (SamePad2d) and
    divide the windowed sum by the windowed count of valid elements. For the
    pools used in MULE (kernel ``[2,1]`` over an even frequency axis) the pad is
    zero, so this reduces to a plain pool — but the mask correction keeps it
    exact for any shape.
    """

    def __init__(self, kernel: tuple[int, int], stride: tuple[int, int]) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.pad = SamePad2d(kernel, stride)

    def forward(self, x: Tensor) -> Tensor:
        kh, kw = self.kernel
        ones = torch.ones_like(x)
        xp = self.pad(x)
        op = self.pad(ones)
        win = (kh * kw)
        summed = F.avg_pool2d(xp, self.kernel, self.stride, padding=0) * win
        count = F.avg_pool2d(op, self.kernel, self.stride, padding=0) * win
        return summed / count


class Stem(nn.Module):
    """A series of weight-standardized convs (scaled-gelu after all but the last)."""

    def __init__(
        self,
        channels: tuple[int, ...],
        kernels: tuple[tuple[int, int], ...],
        strides: tuple[tuple[int, int], ...],
        scaled_activation: str = "gelu",
        in_channels: int = 1,
        bake: bool = False,
    ) -> None:
        super().__init__()
        convs = []
        prev = in_channels
        for c, k, s in zip(channels, kernels, strides):
            convs.append(WSConv2d(prev, c, k, s, groups=1, bake=bake))
            prev = c
        self.convs = nn.ModuleList(convs)
        self.act = ScaledActivation(scaled_activation)
        self.out_channels = prev

    def forward(self, x: Tensor) -> Tensor:
        n = len(self.convs)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < n - 1:  # scaled activation after all but the last conv
                x = self.act(x)
        return x


class NFNetBlock(nn.Module):
    """A single NFNet bottleneck block (sfnfnetf0.py ``_make_nfnet_block``)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: tuple[tuple[int, int], ...],
        freq_downsample: int,
        group_size: int,
        alpha: float,
        beta: float,
        forced_transition: bool = False,
        scaled_activation: str = "gelu",
        bake: bool = False,
    ) -> None:
        super().__init__()
        self.is_transition = (freq_downsample > 1) or (in_channels != out_channels) or forced_transition
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.act = ScaledActivation(scaled_activation)

        bott = out_channels // 2
        groups_mid = bott // group_size
        # 4 residual convs: 1x1->bott, 1x3(group,stride)->bott, 3x1(group)->bott, 1x1->out
        strides = [(1, 1), (freq_downsample, 1), (1, 1), (1, 1)]
        out_chans = [bott, bott, bott, out_channels]
        in_chans = [in_channels, bott, bott, bott]
        groups = [1, groups_mid, groups_mid, 1]
        self.convs = nn.ModuleList(
            [
                WSConv2d(ic, oc, k, s, groups=g, bake=bake)
                for ic, oc, k, s, g in zip(in_chans, out_chans, kernels, strides, groups)
            ]
        )
        self.se = SqueezeExcite(out_channels)
        # Learnable skip-init gain (ScalarMultiply(0.0, learnable=True)); a saved scalar.
        # Shape (1,) (not 0-dim) so it serialises cleanly to safetensors; broadcasts as a scalar.
        self.skip_gain = nn.Parameter(torch.zeros(1))

        # Skip path
        self.skip_pool = TfSameAvgPool2d((freq_downsample, 1), (freq_downsample, 1)) if freq_downsample > 1 else None
        self.skip_conv = (
            WSConv2d(in_channels, out_channels, (1, 1), (1, 1), groups=1, bake=bake)
            if self.is_transition
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.is_transition:
            pre = self.beta * self.act(x)
            skip = pre
            res = pre
        else:
            skip = x
            res = self.beta * self.act(x)

        # Residual path: conv-act x3, conv (no act), SE, skip_gain, alpha.
        res = self.act(self.convs[0](res))
        res = self.act(self.convs[1](res))
        res = self.act(self.convs[2](res))
        res = self.convs[3](res)
        res = self.se(res)
        res = self.skip_gain * res
        res = self.alpha * res

        # Skip path
        if self.skip_pool is not None:
            skip = self.skip_pool(skip)
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        return skip + res


class NFNetStage(nn.Module):
    """A stage = ``num_blocks`` NFNet blocks; block 0 is a (forced) transition."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: tuple[tuple[int, int], ...],
        freq_downsample: int,
        group_size: int,
        alpha: float,
        betas: list[float],
        scaled_activation: str = "gelu",
        bake: bool = False,
    ) -> None:
        super().__init__()
        num_blocks = len(betas)
        blocks = [
            NFNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernels=kernels,
                freq_downsample=freq_downsample,
                group_size=group_size,
                alpha=alpha,
                beta=betas[0],
                forced_transition=True,
                scaled_activation=scaled_activation,
                bake=bake,
            )
        ]
        for i in range(1, num_blocks):
            blocks.append(
                NFNetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernels=kernels,
                    freq_downsample=1,
                    group_size=group_size,
                    alpha=alpha,
                    beta=betas[i],
                    forced_transition=False,
                    scaled_activation=scaled_activation,
                    bake=bake,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class FastToSlowFusion(nn.Module):
    """Fuse the fast path into the slow path (``_make_fast_to_slow_fusion``).

    Time-strided ``1xK`` conv on fast -> ``1x1`` channel expansion -> concat
    onto the slow path along the channel axis.
    """

    def __init__(
        self,
        fast_in_channels: int,
        input_channels: int,
        output_channels: int,
        time_kernel: int = 7,
        time_stride: int = 4,
        bake: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = WSConv2d(fast_in_channels, input_channels, (1, time_kernel), (1, time_stride), groups=1, bake=bake)
        self.conv2 = WSConv2d(input_channels, output_channels, (1, 1), (1, 1), groups=1, bake=bake)

    def forward(self, slow: Tensor, fast: Tensor) -> Tensor:
        fast = self.conv1(fast)
        fast = self.conv2(fast)
        return torch.cat([slow, fast], dim=1)
