"""End-to-end parity vs the original TF MULE.

Gated on converted weights (MULE_TORCH_WEIGHTS) + reference dumps (MULE_TF_REF),
both produced by scripts/convert.py and scripts/verify.py reference. Skipped
otherwise so the rest of the suite runs anywhere.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.requires_weights, pytest.mark.requires_tf_ref]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_clip_embedding_parity(weights_dir, tf_ref_dir):
    from mule_torch import MuleModel

    waveform = np.load(os.path.join(tf_ref_dir, "waveform_16k.npy"))
    ref_timeline = np.load(os.path.join(tf_ref_dir, "timeline.npy"))  # (1728, K)

    model = MuleModel.from_pretrained(model_dir=weights_dir).eval()
    with torch.no_grad():
        torch_clip = model(torch.from_numpy(waveform).float().unsqueeze(0))[0].numpy()
    ref_clip = ref_timeline.mean(axis=1)
    assert _cos(torch_clip, ref_clip) >= 0.9999


def test_backbone_on_reference_slices(weights_dir, tf_ref_dir):
    from mule_torch import MuleModel

    ref_slices = np.load(os.path.join(tf_ref_dir, "slices.npy"))      # (N,96,300,1)
    ref_slice_emb = np.load(os.path.join(tf_ref_dir, "slice_emb.npy"))  # (N,1728)
    model = MuleModel.from_pretrained(model_dir=weights_dir).eval()
    x = torch.from_numpy(np.transpose(ref_slices, (0, 3, 1, 2))).float()
    with torch.no_grad():
        out = model.backbone(x).numpy()
    cos = np.mean([_cos(out[i], ref_slice_emb[i]) for i in range(len(out))])
    assert cos >= 0.9999
