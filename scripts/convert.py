#!/usr/bin/env python
"""Convert the original MULE Keras model (``model.keras``) to mule-torch weights.

Run this ONCE in the ``[convert]`` environment (TensorFlow 2.13, Python <=3.11)::

    python scripts/convert.py \
        --keras  references/music-audio-representations/supporting_data/model/model.keras \
        --references references/music-audio-representations \
        --out artifacts

It produces ``artifacts/model.safetensors`` + ``artifacts/config.json`` (loadable
via ``MuleModel.from_pretrained(model_dir="artifacts")``) and a human-readable
``artifacts/keras_layers.txt`` dump for auditing.

Strategy (see repo README / plan):
  * Build the torch ``SfNfNetF0`` and capture its weighted leaf modules in
    *execution order* via forward hooks (WSConv2d, SqueezeExcite, skip_gain).
  * Walk the Keras model's layers, classify each weighted layer by type, and
    extract its tensors (disambiguated by ndim, so [v, gain] order is irrelevant).
  * Zip per type, asserting shape-compatibility at every step (the safety net
    for any layer-ordering / grouped-conv surprises).
  * Build the librosa mel filterbank and bundle it into the safetensors.

Nothing here trusts a hard-coded layer name; it relies on the deterministic
architecture + shape assertions, and the result is verified numerically by
scripts/verify.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Torch side: build model + capture weighted modules in execution order
# --------------------------------------------------------------------------- #
def capture_torch_order(model):
    """Return ordered lists of (ws_convs, se_blocks, skip_gain_blocks) by exec order."""
    from mule_torch.blocks import NFNetBlock
    from mule_torch.layers import SqueezeExcite, WSConv2d

    ws: list = []
    se: list = []
    skip_blocks: list = []
    handles = []

    def mk(lst):
        def hook(mod, inp, out):
            lst.append(mod)
        return hook

    for m in model.modules():
        if isinstance(m, WSConv2d):
            handles.append(m.register_forward_hook(mk(ws)))
        elif isinstance(m, SqueezeExcite):
            handles.append(m.register_forward_hook(mk(se)))
        elif isinstance(m, NFNetBlock):
            handles.append(m.register_forward_hook(mk(skip_blocks)))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 96, 300))
    for h in handles:
        h.remove()
    return ws, se, skip_blocks


# --------------------------------------------------------------------------- #
# Keras side: classify weighted layers
# --------------------------------------------------------------------------- #
def classify_keras(model):
    """Walk model.layers; return ordered (ws, dense, scalars) extracted as numpy."""
    ws: list[tuple[np.ndarray, np.ndarray]] = []  # (v 4D, gain 1D)
    dense: list[tuple[np.ndarray, np.ndarray]] = []  # (kernel 2D, bias 1D)
    scalars: list[np.ndarray] = []  # learnable ScalarMultiply gain
    table: list[str] = []

    for lyr in model.layers:
        cls = type(lyr).__name__
        weights = lyr.get_weights()
        shapes = [tuple(w.shape) for w in weights]
        table.append(f"{lyr.name:<40s} {cls:<28s} {shapes}")
        if not weights:
            continue
        if cls == "WeightStandardization":
            v = next(w for w in weights if w.ndim == 4)
            gain = next(w for w in weights if w.ndim == 1)
            ws.append((v, gain))
        elif cls == "Dense":
            kernel = next(w for w in weights if w.ndim == 2)
            bias = next((w for w in weights if w.ndim == 1), None)
            if bias is None:
                raise ValueError(f"Dense layer {lyr.name} has no bias; unexpected for SE blocks")
            dense.append((kernel, bias))
        elif cls == "ScalarMultiply":
            # Only learnable ScalarMultiply has a saved weight.
            scal = next((w for w in weights if w.ndim == 0 or w.size == 1), None)
            if scal is not None:
                scalars.append(np.reshape(scal, (1,)))
        else:
            raise ValueError(
                f"Unexpected WEIGHTED Keras layer {lyr.name!r} of type {cls!r} with shapes "
                f"{shapes}. The released embedding model should only contain "
                f"WeightStandardization, Dense (SE), and learnable ScalarMultiply layers. "
                f"If this is a projector/fc/sigmoid training tail, the Keras graph must be "
                f"truncated at the 1728-d embedding node before conversion.",
            )
    return ws, dense, scalars, table


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
def assign_ws(torch_ws, keras_ws):
    if len(torch_ws) != len(keras_ws):
        raise RuntimeError(f"WS conv count mismatch: torch {len(torch_ws)} vs keras {len(keras_ws)}")
    for i, (tmod, (v, gain)) in enumerate(zip(torch_ws, keras_ws)):
        # TF kernel (kH, kW, in/groups, out) -> torch (out, in/groups, kH, kW)
        w = np.transpose(v, (3, 2, 0, 1))
        if tuple(w.shape) != tuple(tmod.weight.shape):
            raise RuntimeError(
                f"WS[{i}] kernel shape mismatch: keras->{w.shape} vs torch {tuple(tmod.weight.shape)}",
            )
        if tuple(gain.shape) != tuple(tmod.gain.shape):
            raise RuntimeError(f"WS[{i}] gain shape mismatch: keras {gain.shape} vs torch {tuple(tmod.gain.shape)}")
        with torch.no_grad():
            tmod.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)).float())
            tmod.gain.copy_(torch.from_numpy(np.ascontiguousarray(gain)).float())


def assign_se(torch_se, keras_dense):
    # Each SE block consumes two consecutive Dense layers (fc1 relu, fc2 sigmoid).
    if len(keras_dense) != 2 * len(torch_se):
        raise RuntimeError(f"Dense count {len(keras_dense)} != 2*SE {2 * len(torch_se)}")
    for i, se in enumerate(torch_se):
        (k1, b1), (k2, b2) = keras_dense[2 * i], keras_dense[2 * i + 1]
        for lin, k, b in ((se.fc1, k1, b1), (se.fc2, k2, b2)):
            wt = np.transpose(k, (1, 0))  # (in,out)->(out,in)
            if tuple(wt.shape) != tuple(lin.weight.shape):
                raise RuntimeError(f"SE[{i}] dense shape mismatch: keras->{wt.shape} vs torch {tuple(lin.weight.shape)}")
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(wt)).float())
                lin.bias.copy_(torch.from_numpy(np.ascontiguousarray(b)).float())


def assign_skip_gains(torch_blocks, keras_scalars):
    if len(torch_blocks) != len(keras_scalars):
        raise RuntimeError(f"skip_gain count mismatch: torch {len(torch_blocks)} vs keras {len(keras_scalars)}")
    for i, (blk, scal) in enumerate(zip(torch_blocks, keras_scalars)):
        with torch.no_grad():
            blk.skip_gain.copy_(torch.from_numpy(np.ascontiguousarray(scal)).float().reshape(1))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MULE Keras weights to mule-torch safetensors.")
    ap.add_argument("--keras", required=True, help="Path to model.keras")
    ap.add_argument("--references", required=True, help="Path to music-audio-representations checkout (for `import mule`)")
    ap.add_argument("--out", default="artifacts", help="Output dir for model.safetensors + config.json")
    ap.add_argument("--bake", action="store_true", help="Pre-fold weight standardization (inference-only, smaller ONNX)")
    args = ap.parse_args()

    # Make the vendored `mule` package importable so custom_objects load.
    sys.path.insert(0, str(Path(args.references).resolve()))
    import tensorflow as tf  # noqa: E402
    import mule.models.layers.activations  # noqa: F401,E402  (registers scaled activations)
    from mule.models.layers import ScalarMultiply, StochDepth, WeightStandardization  # noqa: E402

    from mule_torch.config import MuleConfig  # noqa: E402
    from mule_torch.frontend import build_mel_filterbank  # noqa: E402
    from mule_torch.model import MuleModel  # noqa: E402

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] loading Keras model from {args.keras}")
    custom = {c.__name__: c for c in (WeightStandardization, ScalarMultiply, StochDepth)}
    kmodel = tf.keras.models.load_model(args.keras, custom_objects=custom, compile=False)

    # Dump the layer table first (authoritative for auditing / R1,R3).
    summary_lines: list[str] = []
    kmodel.summary(print_fn=summary_lines.append)
    (out_dir / "keras_summary.txt").write_text("\n".join(summary_lines))
    print(f"[convert] output shape: {kmodel.output_shape}")

    keras_ws, keras_dense, keras_scalars, table = classify_keras(kmodel)
    (out_dir / "keras_layers.txt").write_text("\n".join(table))
    print(f"[convert] keras: {len(keras_ws)} WS-conv, {len(keras_dense)} Dense, {len(keras_scalars)} learnable scalars")

    # Build torch model + capture execution order.
    cfg = MuleConfig(weight_standardization=not args.bake)
    model = MuleModel(cfg, mel_fb=build_mel_filterbank(cfg))
    torch_ws, torch_se, torch_blocks = capture_torch_order(model.backbone)
    print(f"[convert] torch: {len(torch_ws)} WS-conv, {len(torch_se)} SE, {len(torch_blocks)} blocks")

    if args.bake:
        raise SystemExit("--bake conversion path not yet implemented; convert with WS first, bake in a follow-up.")

    # Assign.
    assign_ws(torch_ws, keras_ws)
    assign_se(torch_se, keras_dense)
    assign_skip_gains(torch_blocks, keras_scalars)

    # Self-checks.
    n_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"[convert] backbone params: {n_params/1e6:.2f}M (paper: ~62.4M)")

    # Global parity smoke test: random standardized slice through both nets.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 96, 300, 1)).astype("float32")
    with torch.no_grad():
        torch_out = model.backbone(torch.from_numpy(np.transpose(x, (0, 3, 1, 2)))).numpy()
    keras_out = kmodel.predict(x, verbose=0)
    if keras_out.shape != torch_out.shape:
        print(f"[convert][warn] output shape differs: keras {keras_out.shape} vs torch {torch_out.shape}")
    else:
        max_abs = float(np.max(np.abs(keras_out - torch_out)))
        cos = float(
            np.mean(
                np.sum(keras_out * torch_out, axis=1)
                / (np.linalg.norm(keras_out, axis=1) * np.linalg.norm(torch_out, axis=1) + 1e-12)
            )
        )
        print(f"[convert] smoke parity: max_abs={max_abs:.3e}  mean_cos={cos:.6f}")

    model.save_pretrained(out_dir)
    print(f"[convert] wrote {out_dir/'model.safetensors'} + {out_dir/'config.json'}")


if __name__ == "__main__":
    main()
