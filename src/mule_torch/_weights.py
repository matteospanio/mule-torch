"""Resolve + load MULE weights (safetensors) and config (json).

Supports three sources, in priority order:
1. an explicit local ``model_dir`` containing ``model.safetensors`` + ``config.json``;
2. a Hugging Face repo id (downloaded via ``huggingface_hub``);
3. (fallback) raising a clear error telling the user how to convert weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


def _resolve_local_dir(model_dir: str | os.PathLike | None) -> Path | None:
    if model_dir is not None:
        p = Path(model_dir).expanduser()
        if (p / WEIGHTS_NAME).exists() and (p / CONFIG_NAME).exists():
            return p
        raise FileNotFoundError(
            f"model_dir {p} must contain both {WEIGHTS_NAME} and {CONFIG_NAME}",
        )
    env = os.environ.get("MULE_TORCH_DIR")
    if env:
        p = Path(env).expanduser()
        if (p / WEIGHTS_NAME).exists() and (p / CONFIG_NAME).exists():
            return p
    return None


def load_config_and_state(
    hf_repo: str | None = None,
    revision: str | None = None,
    model_dir: str | os.PathLike | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Return ``(config_dict, state_dict)`` for the requested weights source."""
    local = _resolve_local_dir(model_dir)
    if local is not None:
        cfg = json.loads((local / CONFIG_NAME).read_text())
        state = load_file(str(local / WEIGHTS_NAME))
        return cfg, state

    if hf_repo is not None:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(hf_repo, CONFIG_NAME, revision=revision)
        w_path = hf_hub_download(hf_repo, WEIGHTS_NAME, revision=revision)
        cfg = json.loads(Path(cfg_path).read_text())
        state = load_file(w_path)
        return cfg, state

    raise FileNotFoundError(
        "Could not locate MULE-torch weights. Provide model_dir=<dir with "
        f"{WEIGHTS_NAME}+{CONFIG_NAME}>, set $MULE_TORCH_DIR, or pass hf_repo. "
        "To create the weights, run scripts/convert.py against the original "
        "model.keras (see README).",
    )
