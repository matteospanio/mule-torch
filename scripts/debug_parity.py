#!/usr/bin/env python
"""Localize the first layer where the torch port diverges from Keras.

  # TF venv:    python scripts/debug_parity.py tf    --references REF --keras KERAS --npz artifacts/weights.npz --out artifacts/dbg.npz
  # torch venv: python scripts/debug_parity.py torch --dbg artifacts/dbg.npz --weights artifacts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Keras layer name -> torch backbone return_stages key.
PROBES = [
    ("weight_standardization_3", "slow_stem"),
    ("weight_standardization_7", "fast_stem"),
    ("concatenate", "fuse0"),
    ("stoch_depth", "slow_stage0"),
    ("stoch_depth_12", "fast_stage0"),
    ("concatenate_1", "fuse1"),
    ("concatenate_2", "fuse2"),
    ("concatenate_3", "fuse3"),
]


def run_tf(args):
    sys.path.insert(0, str(Path(args.references).resolve()))
    import tensorflow as tf
    import mule.models.layers.activations  # noqa: F401
    from mule.models.layers import ScalarMultiply, StochDepth, WeightStandardization

    custom = {c.__name__: c for c in (WeightStandardization, ScalarMultiply, StochDepth)}
    m = tf.keras.models.load_model(args.keras, custom_objects=custom, compile=False)
    smoke_x = np.load(args.npz)["smoke_x"]  # (2,96,300,1)
    names = [n for n, _ in PROBES]
    sub = tf.keras.Model(m.input, [m.get_layer(n).output for n in names])
    outs = sub.predict(smoke_x, verbose=0)
    store = {"smoke_x": smoke_x}
    for (n, label), arr in zip(PROBES, outs):
        store[n] = np.asarray(arr, dtype="float32")
        print(f"[tf] {label:<26s} {n:<26s} shape {arr.shape}")
    np.savez(args.out, **store)
    print(f"[tf] wrote {args.out}")


def _cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_torch(args):
    import torch

    from mule_torch import MuleModel

    dbg = np.load(args.dbg)
    x = torch.from_numpy(np.transpose(dbg["smoke_x"], (0, 3, 1, 2))).float()  # NCHW
    model = MuleModel.from_pretrained(model_dir=args.weights).eval()
    bb = model.backbone

    def cmp(name, label, torch_nchw):
        ref = dbg[name]  # NHWC
        t = torch_nchw.permute(0, 2, 3, 1).detach().numpy()  # -> NHWC
        k = min(t.shape[1], ref.shape[1])
        kk = min(t.shape[2], ref.shape[2]) if t.ndim == 4 else None
        if t.shape != ref.shape:
            print(f"[torch] {label:<26s} SHAPE DIFF torch {t.shape} vs tf {ref.shape}")
            return
        print(f"[torch] {label:<26s} max_abs={np.abs(t-ref).max():.3e} cos={_cos(t, ref):.6f}")

    with torch.no_grad():
        _, st = bb(x, return_stages=True)
        for kname, skey in PROBES:
            if skey in st:
                cmp(kname, skey, st[skey])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tf")
    t.add_argument("--references", required=True)
    t.add_argument("--keras", required=True)
    t.add_argument("--npz", default="artifacts/weights.npz")
    t.add_argument("--out", default="artifacts/dbg.npz")
    p = sub.add_parser("torch")
    p.add_argument("--dbg", default="artifacts/dbg.npz")
    p.add_argument("--weights", default="artifacts")
    args = ap.parse_args()
    run_tf(args) if args.mode == "tf" else run_torch(args)


if __name__ == "__main__":
    main()
