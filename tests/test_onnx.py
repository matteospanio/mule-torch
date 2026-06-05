"""ONNX export + onnxruntime parity for the backbone (random init; no weights needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mule_torch.backbone import SfNfNetF0

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def test_backbone_onnx_parity(tmp_path):
    model = SfNfNetF0().eval()
    dummy = torch.randn(1, 1, 96, 300)
    onnx_path = tmp_path / "backbone.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["mel_slice"],
        output_names=["embedding"],
        dynamic_axes={"mel_slice": {0: "n"}, "embedding": {0: "n"}},
        opset_version=17,
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Use a different batch size to exercise the dynamic axis.
    x = torch.randn(3, 1, 96, 300)
    onnx_out = sess.run(None, {"mel_slice": x.numpy()})[0]
    with torch.no_grad():
        ref = model(x).numpy()
    assert onnx_out.shape == ref.shape == (3, 1728)
    assert np.abs(onnx_out - ref).max() < 1e-4
