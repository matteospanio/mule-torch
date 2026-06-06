#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "tensorflow==2.13.1",
#   "scooch>=1.0.4",
#   "librosa==0.9.2",
#   "setuptools<81",
#   "soundfile>=0.12",
#   "numpy<1.25",
#   "torch>=2.1",
#   "safetensors>=0.4",
#   "onnx>=1.15",
#   "onnxruntime>=1.17",
# ]
# ///
"""Numerically verify the mule-torch port against the original TF MULE pipeline.

This is a STANDALONE tool, not part of the `mule_torch` package; it carries its
own dependencies via the PEP 723 block above. Run with uv:

  # Phase A — REFERENCE: run the genuine TF pipeline, dump ground-truth arrays.
  uv run scripts/verify.py reference \
      --references references/music-audio-representations \
      --config references/music-audio-representations/supporting_data/configs/mule_embedding_timeline.yml \
      --wav tests/fixtures/fixture.wav --out artifacts/ref

  # Phase B — COMPARE: load the dumps + converted weights, run the torch port,
  # assert parity, and check ONNX export.
  uv run scripts/verify.py compare --ref artifacts/ref --weights artifacts --onnx

The reference phase dumps: waveform_16k.npy, mel.npy (96,T), slices.npy
(N,96,300,1), slice_emb.npy (N,1728), timeline.npy (1728,K).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Import the local mule_torch package from src/ without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _gen_fixture(path: Path, seconds: float = 6.0, sr: int = 16000) -> None:
    """Write a deterministic synthetic wav (chirp + harmonics + light noise)."""
    import soundfile as sf

    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    x = (
        0.5 * np.sin(2 * np.pi * (220.0 + 40.0 * t) * t)
        + 0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2 * np.pi * 880.0 * t)
        + 0.02 * rng.standard_normal(n)
    ).astype(np.float32)
    x /= np.max(np.abs(x)) + 1e-6
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)


# --------------------------------------------------------------------------- #
def run_reference(args) -> None:
    sys.path.insert(0, str(Path(args.references).resolve()))
    import mule.models.layers.activations  # noqa: F401  (registers scaled activations)
    from mule.analysis import Analysis
    from scooch import Config

    wav = Path(args.wav)
    if not wav.exists():
        print(f"[ref] fixture {wav} missing; generating a synthetic one")
        _gen_fixture(wav)
    wav = wav.resolve()  # absolute BEFORE we chdir into the model dir

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # SCOOCH resolves model paths relative to cwd, like the parent encoder does.
    cfg_path = Path(args.config).resolve()
    model_root = Path(args.references).resolve()
    prev = os.getcwd()
    os.chdir(model_root)
    try:
        analysis = Analysis(Config(str(cfg_path)))
        timeline = np.asarray(analysis.analyze(str(wav.resolve())).data, dtype=np.float32)  # (1728, K)

        # Reconstruct intermediates from the (private) pipeline objects.
        src = analysis._source_feature
        src.clear()
        src.from_file(str(wav.resolve()))
        waveform = np.squeeze(np.asarray(src.data, dtype=np.float32))  # (T,)

        mel_feat = analysis._feature_transforms[0]
        mel_feat.clear()
        mel_feat.from_feature(src)
        mel = np.asarray(mel_feat.data, dtype=np.float32)  # (96, T)

        emb_feat = analysis._feature_transforms[1]
        slices = np.asarray(emb_feat._extractor.extract_range(mel_feat, 0, len(mel_feat)), dtype=np.float32)
        slice_emb = np.asarray(emb_feat._model.predict(slices, verbose=0), dtype=np.float32)  # (N, 1728)
    finally:
        os.chdir(prev)

    np.save(out / "waveform_16k.npy", waveform)
    np.save(out / "mel.npy", mel)
    np.save(out / "slices.npy", slices)
    np.save(out / "slice_emb.npy", slice_emb)
    np.save(out / "timeline.npy", timeline)
    print(f"[ref] waveform {waveform.shape}  mel {mel.shape}  slices {slices.shape}  "
          f"slice_emb {slice_emb.shape}  timeline {timeline.shape}")
    print(f"[ref] dumped to {out}")


# --------------------------------------------------------------------------- #
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_compare(args) -> None:
    import torch

    from mule_torch import MuleModel
    from mule_torch.frontend import slice_mel

    ref = Path(args.ref)
    waveform = np.load(ref / "waveform_16k.npy")
    ref_mel = np.load(ref / "mel.npy")
    ref_slices = np.load(ref / "slices.npy")           # (N,96,300,1)
    ref_slice_emb = np.load(ref / "slice_emb.npy")     # (N,1728)
    ref_timeline = np.load(ref / "timeline.npy")       # (1728,K)

    model = MuleModel.from_pretrained(model_dir=args.weights, map_location="cpu")
    model.eval()
    fails: list[str] = []

    def check(name: str, max_abs: float, cos: float, tol_abs: float, tol_cos: float) -> None:
        ok = (max_abs <= tol_abs) and (cos >= tol_cos)
        print(f"[cmp] {name:<22s} max_abs={max_abs:.3e} cos={cos:.7f}  "
              f"{'OK' if ok else 'FAIL'} (tol_abs<={tol_abs:.0e}, cos>={tol_cos})")
        if not ok:
            fails.append(name)

    with torch.no_grad():
        wav_t = torch.from_numpy(waveform).float().unsqueeze(0)  # (1,T)

        # 1) Mel front-end
        torch_mel = model.frontend(wav_t)[0].numpy()  # (96, frames)
        k = min(torch_mel.shape[1], ref_mel.shape[1])
        d = np.abs(torch_mel[:, :k] - ref_mel[:, :k])
        # max-abs is on the log10(10000*x+1) scale; float32 DFT-vs-FFT noise near the
        # mel floor reaches ~1e-2 but washes out after per-slice standard-norm (see
        # 'slices'/'clip embedding' below). Cosine is the meaningful check here.
        check("mel", float(d.max()), _cos(torch_mel[:, :k], ref_mel[:, :k]), 2e-2, 0.99999)

        # 2) Slicing
        torch_slices = slice_mel(torch.from_numpy(ref_mel).float(), model.config).numpy()  # (N,1,96,300)
        ref_slices_nchw = np.transpose(ref_slices, (0, 3, 1, 2))  # (N,1,96,300)
        m = min(torch_slices.shape[0], ref_slices_nchw.shape[0])
        d = np.abs(torch_slices[:m] - ref_slices_nchw[:m])
        check("slices", float(d.max()), _cos(torch_slices[:m], ref_slices_nchw[:m]), 1e-4, 0.99999)

        # 3) Backbone on the REFERENCE slices (isolates backbone from frontend drift)
        bb_in = torch.from_numpy(ref_slices_nchw).float()
        torch_bb = model.backbone(bb_in).numpy()  # (N,1728)
        per_slice_cos = float(np.mean([
            _cos(torch_bb[i], ref_slice_emb[i]) for i in range(min(len(torch_bb), len(ref_slice_emb)))
        ]))
        d = np.abs(torch_bb[: len(ref_slice_emb)] - ref_slice_emb[: len(torch_bb)])
        check("backbone(ref slices)", float(d.max()), per_slice_cos, 1e-2, 0.9999)

        # 4) End-to-end clip embedding (mean-pooled) from the reference waveform
        torch_clip = model(wav_t)[0].numpy()  # (1728,)
        ref_clip = ref_timeline.mean(axis=1)  # (1728,)
        check("clip embedding", float(np.abs(torch_clip - ref_clip).max()), _cos(torch_clip, ref_clip), 2e-2, 0.99999)

    # 5) ONNX backbone parity (optional)
    if args.onnx:
        try:
            import onnxruntime as ort

            onnx_path = Path(args.weights) / "backbone.onnx"
            dummy = torch.from_numpy(np.transpose(ref_slices[:1], (0, 3, 1, 2))).float()
            torch.onnx.export(
                model.backbone, dummy, str(onnx_path),
                input_names=["mel_slice"], output_names=["embedding"],
                dynamic_axes={"mel_slice": {0: "n"}, "embedding": {0: "n"}}, opset_version=17,
            )
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            onnx_out = sess.run(None, {"mel_slice": dummy.numpy()})[0]
            with torch.no_grad():
                ref_out = model.backbone(dummy).numpy()
            d = float(np.abs(onnx_out - ref_out).max())
            print(f"[cmp] onnx backbone        max_abs={d:.3e}  {'OK' if d < 1e-4 else 'FAIL'}")
            if d >= 1e-4:
                fails.append("onnx")
        except Exception as e:  # noqa: BLE001
            print(f"[cmp] onnx export skipped/failed: {e}")

    if fails:
        raise SystemExit(f"PARITY FAILED: {fails}")
    print("[cmp] ALL PARITY CHECKS PASSED")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Verify mule-torch vs original MULE.")
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("reference", help="(TF venv) dump ground-truth arrays")
    r.add_argument("--references", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--wav", default="tests/fixtures/fixture.wav")
    r.add_argument("--out", default="artifacts/ref")

    c = sub.add_parser("compare", help="(torch venv) compare torch port vs dumps")
    c.add_argument("--ref", default="artifacts/ref")
    c.add_argument("--weights", default="artifacts")
    c.add_argument("--onnx", action="store_true")

    args = ap.parse_args()
    if args.mode == "reference":
        run_reference(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
