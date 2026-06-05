"""``MuleModel`` — the full waveform -> 1728-d embedding pipeline.

This is the public entry point the parent project's encoder seam expects::

    from mule_torch import MuleModel
    model = MuleModel.from_pretrained(model_dir="...")   # or hf_repo="csc-unipd/mule-torch"
    emb = model(waveform)   # (B, T) @ 16 kHz  ->  (B, 1728)

``forward`` runs, per clip: mel front-end -> 96x300 slices every ~2 s ->
SF-NFNet-F0 backbone -> mean over the slice timeline -> one 1728-d vector.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor, nn

from mule_torch._weights import load_config_and_state
from mule_torch.backbone import SfNfNetF0
from mule_torch.config import MuleConfig
from mule_torch.frontend import MuleMelSpectrogram, slice_mel


class MuleModel(nn.Module):
    """Frontend + SF-NFNet-F0 backbone + 2 s-timeline mean-pool."""

    def __init__(self, config: MuleConfig | None = None, mel_fb: Tensor | None = None) -> None:
        super().__init__()
        cfg = config or MuleConfig()
        self.config = cfg
        self.frontend = MuleMelSpectrogram(cfg, mel_fb=mel_fb)
        self.backbone = SfNfNetF0(cfg)

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def embed_timeline(self, waveform: Tensor) -> list[Tensor]:
        """Per-clip timeline of slice embeddings: list of ``(N_b, 1728)`` tensors."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mels = self.frontend(waveform)  # (B, n_mels, frames)
        out: list[Tensor] = []
        for b in range(mels.shape[0]):
            slices = slice_mel(mels[b], self.config)  # (N, 1, n_mels, width)
            out.append(self.backbone(slices))         # (N, 1728)
        return out

    def forward(self, waveform: Tensor) -> Tensor:
        """``(B, T)`` waveform @ 16 kHz -> ``(B, 1728)`` mean-pooled clip embedding."""
        timeline = self.embed_timeline(waveform)
        return torch.stack([t.mean(dim=0) for t in timeline], dim=0)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str | None = None,
        revision: str | None = None,
        model_dir: str | os.PathLike | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "MuleModel":
        cfg_dict, state = load_config_and_state(hf_repo=hf_repo, revision=revision, model_dir=model_dir)
        cfg = MuleConfig.from_dict(cfg_dict)
        # The mel filterbank is stored in the state dict; build the model with it
        # so MuleMelSpectrogram does not try to import librosa at inference time.
        mel_fb = state.get("frontend.mel_fb")
        model = cls(cfg, mel_fb=mel_fb)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # STFT DFT buffers are deterministic and may be absent from older dumps;
        # tolerate only those as "missing". Anything else is a real error.
        real_missing = [k for k in missing if not k.startswith("frontend.stft.")]
        if strict and (real_missing or unexpected):
            raise RuntimeError(
                f"state_dict mismatch. missing={real_missing} unexpected={list(unexpected)}",
            )
        model.to(map_location)
        model.eval()
        return model

    def save_pretrained(self, save_dir: str | os.PathLike) -> None:
        """Write ``config.json`` + ``model.safetensors`` (used by convert.py)."""
        import json
        from pathlib import Path

        from safetensors.torch import save_file

        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        state = {k: v.contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state, str(d / "model.safetensors"))
