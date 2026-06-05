#!/usr/bin/env python
"""Step through slow-stage0 block0 and compare every internal to Keras.

  # TF:    python scripts/debug_block.py tf    --references REF --keras KERAS --npz artifacts/weights.npz --out artifacts/dbgb.npz
  # torch: python scripts/debug_block.py torch --dbg artifacts/dbgb.npz --weights artifacts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# slow stage0 block0 internals (creation-order names): conv0..3 raw, skip conv raw, block out.
PROBES = ["concatenate", "weight_standardization_8", "weight_standardization_9",
          "weight_standardization_10", "weight_standardization_11", "weight_standardization_12",
          "stoch_depth"]


def run_tf(args):
    sys.path.insert(0, str(Path(args.references).resolve()))
    import tensorflow as tf
    import mule.models.layers.activations  # noqa: F401
    from mule.models.layers import ScalarMultiply, StochDepth, WeightStandardization

    custom = {c.__name__: c for c in (WeightStandardization, ScalarMultiply, StochDepth)}
    m = tf.keras.models.load_model(args.keras, custom_objects=custom, compile=False)
    smoke_x = np.load(args.npz)["smoke_x"]
    sub = tf.keras.Model(m.input, [m.get_layer(n).output for n in PROBES])
    outs = sub.predict(smoke_x, verbose=0)
    store = {n: np.asarray(o, dtype="float32") for n, o in zip(PROBES, outs)}
    np.savez(args.out, **store)
    for n, o in zip(PROBES, outs):
        print(f"[tf] {n:<26s} {o.shape}")


def _stats(label, t_nchw, ref_nhwc):
    t = t_nchw.permute(0, 2, 3, 1).detach().numpy()
    if t.shape != ref_nhwc.shape:
        print(f"[torch] {label:<22s} SHAPE {t.shape} vs {ref_nhwc.shape}")
        return
    a, b = t.reshape(-1), ref_nhwc.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"[torch] {label:<22s} max_abs={np.abs(t-ref_nhwc).max():.3e} cos={cos:.6f}")


def run_torch(args):
    import torch

    from mule_torch import MuleModel

    dbg = np.load(args.dbg)
    model = MuleModel.from_pretrained(model_dir=args.weights).eval()
    blk = model.backbone.slow_stages[0].blocks[0]
    fuse0 = torch.from_numpy(np.transpose(dbg["concatenate"], (0, 3, 1, 2))).float()

    with torch.no_grad():
        pre = blk.beta * blk.act(fuse0)
        c0 = blk.convs[0](pre); _stats("conv0", c0, dbg["weight_standardization_8"])
        a0 = blk.act(c0)
        c1 = blk.convs[1](a0); _stats("conv1", c1, dbg["weight_standardization_9"])
        a1 = blk.act(c1)
        c2 = blk.convs[2](a1); _stats("conv2", c2, dbg["weight_standardization_10"])
        a2 = blk.act(c2)
        c3 = blk.convs[3](a2); _stats("conv3", c3, dbg["weight_standardization_11"])
        skip = blk.skip_conv(pre); _stats("skip_conv", skip, dbg["weight_standardization_12"])
        out = blk(fuse0); _stats("block_out", out, dbg["stoch_depth"])
        print(f"[torch] beta={blk.beta} alpha={blk.alpha} skip_gain={float(blk.skip_gain):.6f}")

        # Decompose residual = block_out - skip ; SE = residual / (alpha*skip_gain)
        skip_k = torch.from_numpy(np.transpose(dbg["weight_standardization_12"], (0, 3, 1, 2))).float()
        out_k = torch.from_numpy(np.transpose(dbg["stoch_depth"], (0, 3, 1, 2))).float()
        res_k = out_k - skip_k
        res_t = out - skip
        _stats("residual", res_t, res_k.permute(0, 2, 3, 1).numpy())
        se_t = blk.se(c3)
        scale = blk.alpha * float(blk.skip_gain)
        se_k = res_k / scale
        _stats("SE_out", se_t, se_k.permute(0, 2, 3, 1).numpy())
        # SE gate (per-channel) torch:
        g = torch.sigmoid(blk.se.fc2(torch.relu(blk.se.fc1(c3.mean(dim=(2, 3)))))) * 2.0
        print(f"[torch] SE gate mean={float(g.mean()):.4f} min={float(g.min()):.4f} max={float(g.max()):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tf")
    t.add_argument("--references", required=True)
    t.add_argument("--keras", required=True)
    t.add_argument("--npz", default="artifacts/weights.npz")
    t.add_argument("--out", default="artifacts/dbgb.npz")
    p = sub.add_parser("torch")
    p.add_argument("--dbg", default="artifacts/dbgb.npz")
    p.add_argument("--weights", default="artifacts")
    args = ap.parse_args()
    run_tf(args) if args.mode == "tf" else run_torch(args)


if __name__ == "__main__":
    main()
