#!/usr/bin/env python
"""Convert the original MULE Keras model to mule-torch weights, in two phases.

TensorFlow and torch are decoupled into separate venvs (tf2.13 + torch + librosa
in one env is a dependency-resolution minefield), with a numpy ``.npz`` as the
interchange:

  # Phase 1 — EXTRACT  (TF venv, Python <=3.11): keras -> weights.npz
  python scripts/convert.py extract \
      --keras references/.../supporting_data/model/model.keras \
      --references references/music-audio-representations \
      --out artifacts/weights.npz

  # Phase 2 — ASSEMBLE (torch venv): weights.npz -> model.safetensors + config.json
  python scripts/convert.py assemble --npz artifacts/weights.npz --out artifacts

The extract phase also bundles the *librosa 0.9.2* mel filterbank and a fixed
random-input keras output, so assemble can build a bit-matching frontend and run
an immediate parity smoke test without importing TensorFlow.

Mapping is by per-type execution order with shape assertions at every step (the
safety net for layer-ordering / grouped-conv surprises); verified numerically by
scripts/verify.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ===================== Phase 1: EXTRACT (TensorFlow) ======================= #
def _name_suffix(name: str) -> int:
    """Keras numbers each layer type in CREATION order via a name suffix.

    'weight_standardization' -> 0, 'weight_standardization_47' -> 47. This is the
    robust ordering key: model.layers comes back in a topologically-interleaved
    order, but the creation-order suffix matches the deterministic _make_layers
    construction (slow stem, fast stem, slow stages, fast stages, fusions).
    """
    tail = name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def classify_keras(model):
    """Return creation-ordered (ws, dense, scalars, table).

    Weights are read from each layer's ``.weights`` Variables so gain vs bias can
    be told apart by variable name (both are 1-D), rather than guessed by shape.
    Each WS conv -> (v 4D, gain 1D, bias 1D-or-None); each Dense -> (kernel, bias);
    each learnable ScalarMultiply -> scalar. Lists are sorted by the layer's
    name-suffix so they follow creation order regardless of model.layers order.
    """
    ws: list[tuple[int, tuple]] = []
    dense: list[tuple[int, tuple]] = []
    scalars: list[tuple[int, np.ndarray]] = []
    table: list[str] = []
    for lyr in model.layers:
        cls = type(lyr).__name__
        vars_ = list(lyr.weights)
        shapes = [(v.name, tuple(v.shape)) for v in vars_]
        table.append(f"{_name_suffix(lyr.name):>4d}  {lyr.name:<36s} {cls:<26s} {shapes}")
        if not vars_:
            continue
        sfx = _name_suffix(lyr.name)
        if cls == "WeightStandardization":
            v = gain = bias = None
            for var in vars_:
                arr = var.numpy().astype("float32")
                if arr.ndim == 4:
                    v = arr
                elif "bias" in var.name:
                    bias = arr
                else:
                    gain = arr
            if v is None or gain is None:
                raise ValueError(f"WS layer {lyr.name}: missing kernel/gain in {shapes}")
            ws.append((sfx, (v, gain, bias)))
        elif cls == "Dense":
            kernel = bias = None
            for var in vars_:
                arr = var.numpy().astype("float32")
                if arr.ndim == 2:
                    kernel = arr
                elif "bias" in var.name:
                    bias = arr
            if kernel is None or bias is None:
                raise ValueError(f"Dense {lyr.name}: missing kernel/bias in {shapes}")
            dense.append((sfx, (kernel, bias)))
        elif cls == "ScalarMultiply":
            scalars.append((sfx, np.reshape(vars_[0].numpy(), (1,)).astype("float32")))
        else:
            raise ValueError(
                f"Unexpected WEIGHTED layer {lyr.name!r} ({cls}) {shapes}. The released "
                f"embedding model should only contain WeightStandardization, Dense (SE) and "
                f"learnable ScalarMultiply. A projector/fc/sigmoid training tail means the graph "
                f"must be truncated at the 1728-d embedding node first.",
            )
    ws_sorted = [d for _, d in sorted(ws, key=lambda t: t[0])]
    dense_sorted = [d for _, d in sorted(dense, key=lambda t: t[0])]
    scalars_sorted = [d for _, d in sorted(scalars, key=lambda t: t[0])]
    return ws_sorted, dense_sorted, scalars_sorted, table


def _inbound_layers(layer) -> list:
    node = layer.inbound_nodes[0]
    inb = node.inbound_layers
    return list(inb) if isinstance(inb, (list, tuple)) else [inb]


def extract_skip_gains(model) -> list[np.ndarray]:
    """Map each block's learnable skip-init gain to its block via graph connectivity.

    The learnable ScalarMultiply's *name suffix* does NOT follow block creation
    order (ScalarMultiply creates its Variable before super().__init__, scrambling
    the auto-name counter). So instead we walk each StochDepth (block output, whose
    suffix IS block-ordered) back through its residual branch:
    StochDepth <- ScalarMultiply(alpha) <- ScalarMultiply(learnable skip_gain).
    Returns gains in block creation order (slow stages then fast stages).
    """
    sds = [l for l in model.layers if type(l).__name__ == "StochDepth"]
    sds.sort(key=lambda l: _name_suffix(l.name))
    gains: list[np.ndarray] = []
    for sd in sds:
        sms = [l for l in _inbound_layers(sd) if type(l).__name__ == "ScalarMultiply"]
        if len(sms) != 1:
            raise ValueError(f"{sd.name}: expected 1 ScalarMultiply (alpha) input, got {[type(l).__name__ for l in _inbound_layers(sd)]}")
        alpha_sm = sms[0]
        learn = [l for l in _inbound_layers(alpha_sm) if type(l).__name__ == "ScalarMultiply"]
        if len(learn) != 1:
            raise ValueError(f"{alpha_sm.name}: expected 1 ScalarMultiply (skip_gain) input")
        w = learn[0].weights
        if len(w) != 1:
            raise ValueError(f"{learn[0].name}: expected a single learnable scalar, got {len(w)} weights")
        gains.append(np.reshape(w[0].numpy(), (1,)).astype("float32"))
    return gains


def run_extract(args) -> None:
    sys.path.insert(0, str(Path(args.references).resolve()))
    import tensorflow as tf  # noqa: E402
    import mule.models.layers.activations  # noqa: F401,E402  (registers scaled activations)
    from mule.models.layers import ScalarMultiply, StochDepth, WeightStandardization  # noqa: E402

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[extract] loading {args.keras}")
    custom = {c.__name__: c for c in (WeightStandardization, ScalarMultiply, StochDepth)}
    kmodel = tf.keras.models.load_model(args.keras, custom_objects=custom, compile=False)
    print(f"[extract] keras output shape: {kmodel.output_shape}")

    summary: list[str] = []
    kmodel.summary(print_fn=summary.append)
    out.with_suffix(".summary.txt").write_text("\n".join(summary))

    ws, dense, scalars_unused, table = classify_keras(kmodel)
    out.with_suffix(".layers.txt").write_text("\n".join(table))
    # skip_gains mapped to blocks via graph connectivity (suffix order is unreliable).
    scalars = extract_skip_gains(kmodel)
    print(f"[extract] {len(ws)} WS-conv, {len(dense)} Dense, {len(scalars)} skip_gains "
          f"(suffix-sorted learnable count was {len(scalars_unused)})")

    # librosa 0.9.2 mel filterbank (matches the reference pipeline exactly).
    import librosa

    mel_fb = librosa.filters.mel(
        sr=16000, n_fft=2048, n_mels=96, fmin=0.0, fmax=8000.0, htk=True, norm=2.0
    ).astype("float32")

    # Fixed-input keras output, for an immediate parity smoke test in assemble.
    rng = np.random.default_rng(0)
    smoke_x = rng.standard_normal((2, 96, 300, 1)).astype("float32")
    smoke_keras_out = np.asarray(kmodel.predict(smoke_x, verbose=0), dtype="float32")

    n_bias = sum(1 for _, _, b in ws if b is not None)
    store: dict[str, np.ndarray] = {
        "n_ws": np.array(len(ws)), "n_dense": np.array(len(dense)), "n_scalar": np.array(len(scalars)),
        "ws_has_bias": np.array(int(n_bias > 0)),
        "mel_fb": mel_fb, "smoke_x": smoke_x, "smoke_keras_out": smoke_keras_out,
    }
    print(f"[extract] WS biases present: {n_bias}/{len(ws)}")
    for i, (v, g, b) in enumerate(ws):
        store[f"ws_v_{i}"] = v
        store[f"ws_gain_{i}"] = g
        if b is not None:
            store[f"ws_bias_{i}"] = b
    for i, (k, b) in enumerate(dense):
        store[f"dense_k_{i}"] = k
        store[f"dense_b_{i}"] = b
    for i, s in enumerate(scalars):
        store[f"scalar_{i}"] = s
    np.savez(out, **store)
    print(f"[extract] wrote {out} (+ .layers.txt, .summary.txt)")


# ===================== Phase 2: ASSEMBLE (torch) =========================== #
def creation_order_modules(backbone):
    """Walk the backbone in the SAME order Keras created the layers (_make_layers):
    slow stem, fast stem, slow stages, fast stages, fusions. Within a block:
    conv0..3 then skip_conv; SE denses and skip_gains in block order.

    Returns (ws_convs, se_blocks, blocks) — matching the name-suffix-sorted Keras lists.
    """
    ws: list = list(backbone.slow_stem.convs) + list(backbone.fast_stem.convs)
    se: list = []
    blocks: list = []

    def walk_stages(stages):
        for stage in stages:
            for blk in stage.blocks:
                ws.extend(list(blk.convs))
                if blk.skip_conv is not None:
                    ws.append(blk.skip_conv)
                se.append(blk.se)
                blocks.append(blk)

    walk_stages(backbone.slow_stages)
    walk_stages(backbone.fast_stages)
    for fuse in backbone.fusions:
        ws.extend([fuse.conv1, fuse.conv2])
    return ws, se, blocks


def run_assemble(args) -> None:
    import torch

    from mule_torch.config import MuleConfig
    from mule_torch.model import MuleModel

    data = np.load(args.npz)
    n_ws, n_dense, n_scalar = int(data["n_ws"]), int(data["n_dense"]), int(data["n_scalar"])
    print(f"[assemble] npz: {n_ws} WS-conv, {n_dense} Dense, {n_scalar} scalars")

    cfg = MuleConfig(weight_standardization=not args.bake)
    model = MuleModel(cfg, mel_fb=torch.from_numpy(data["mel_fb"]).float())
    torch_ws, torch_se, torch_blocks = creation_order_modules(model.backbone)
    print(f"[assemble] torch: {len(torch_ws)} WS-conv, {len(torch_se)} SE, {len(torch_blocks)} blocks")

    if args.bake:
        raise SystemExit("--bake not implemented; assemble with WS first.")

    # --- WS convs ---
    assert len(torch_ws) == n_ws, f"WS count {len(torch_ws)} != {n_ws}"
    n_bias_assigned = 0
    for i, tmod in enumerate(torch_ws):
        v = data[f"ws_v_{i}"]          # (kH,kW,in/g,out)
        gain = data[f"ws_gain_{i}"]    # (out,)
        w = np.transpose(v, (3, 2, 0, 1))
        assert tuple(w.shape) == tuple(tmod.weight.shape), f"WS[{i}] {w.shape} vs {tuple(tmod.weight.shape)}"
        assert tuple(gain.shape) == tuple(tmod.gain.shape), f"WS[{i}] gain {gain.shape} vs {tuple(tmod.gain.shape)}"
        with torch.no_grad():
            tmod.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)).float())
            tmod.gain.copy_(torch.from_numpy(np.ascontiguousarray(gain)).float())
            key = f"ws_bias_{i}"
            if key in data.files:
                b = data[key]
                assert tuple(b.shape) == tuple(tmod.bias.shape), f"WS[{i}] bias {b.shape} vs {tuple(tmod.bias.shape)}"
                tmod.bias.copy_(torch.from_numpy(np.ascontiguousarray(b)).float())
                n_bias_assigned += 1
    print(f"[assemble] WS biases assigned: {n_bias_assigned}/{n_ws}")

    # --- SE denses (two consecutive Dense per SE block) ---
    assert n_dense == 2 * len(torch_se), f"Dense {n_dense} != 2*SE {2*len(torch_se)}"
    for i, se in enumerate(torch_se):
        for lin, ki, bi in ((se.fc1, 2 * i, 2 * i), (se.fc2, 2 * i + 1, 2 * i + 1)):
            k = data[f"dense_k_{ki}"]  # (in,out)
            b = data[f"dense_b_{bi}"]
            wt = np.transpose(k, (1, 0))
            assert tuple(wt.shape) == tuple(lin.weight.shape), f"SE[{i}] {wt.shape} vs {tuple(lin.weight.shape)}"
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(wt)).float())
                lin.bias.copy_(torch.from_numpy(np.ascontiguousarray(b)).float())

    # --- skip gains ---
    assert len(torch_blocks) == n_scalar, f"skip_gain {len(torch_blocks)} != {n_scalar}"
    for i, blk in enumerate(torch_blocks):
        with torch.no_grad():
            blk.skip_gain.copy_(torch.from_numpy(np.ascontiguousarray(data[f"scalar_{i}"])).float().reshape(1))

    n_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"[assemble] backbone params: {n_params/1e6:.2f}M (paper ~62.4M)")

    # Smoke parity vs the keras output stored at extract time.
    x = data["smoke_x"]  # (2,96,300,1)
    with torch.no_grad():
        torch_out = model.backbone(torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).float()).numpy()
    keras_out = data["smoke_keras_out"]
    if keras_out.shape == torch_out.shape:
        max_abs = float(np.max(np.abs(keras_out - torch_out)))
        cos = float(np.mean(np.sum(keras_out * torch_out, 1) /
                            (np.linalg.norm(keras_out, axis=1) * np.linalg.norm(torch_out, axis=1) + 1e-12)))
        print(f"[assemble] SMOKE PARITY: max_abs={max_abs:.3e}  mean_cos={cos:.6f}")
    else:
        print(f"[assemble][warn] shape mismatch keras {keras_out.shape} vs torch {torch_out.shape}")

    out = Path(args.out)
    model.save_pretrained(out)
    print(f"[assemble] wrote {out/'model.safetensors'} + {out/'config.json'}")


# =========================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MULE Keras -> mule-torch (two phases).")
    sub = ap.add_subparsers(dest="phase", required=True)

    e = sub.add_parser("extract", help="(TF venv) keras -> weights.npz")
    e.add_argument("--keras", required=True)
    e.add_argument("--references", required=True)
    e.add_argument("--out", default="artifacts/weights.npz")

    a = sub.add_parser("assemble", help="(torch venv) weights.npz -> safetensors")
    a.add_argument("--npz", default="artifacts/weights.npz")
    a.add_argument("--out", default="artifacts")
    a.add_argument("--bake", action="store_true")

    args = ap.parse_args()
    if args.phase == "extract":
        run_extract(args)
    else:
        run_assemble(args)


if __name__ == "__main__":
    main()
