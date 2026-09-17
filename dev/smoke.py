"""Task-author dev checks. Not shipped into any container.

    python dev/smoke.py parity    # tiny config: starter vs oracle must agree
    python dev/smoke.py budget    # full config: starter must OOM, oracle must fit

`parity` is the important one -- it proves the two implementations compute the
same thing before the VRAM numbers mean anything.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.join(HERE, "..", "tasks", "wan-vram-budget")
STARTER = os.path.join(TASK, "environment", "app", "pipeline.py")
ORACLE = os.path.join(TASK, "tests", "reference", "oracle_pipeline.py")

GIB = 1024**3


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_frames(out_dir: str, shape):
    raw = open(os.path.join(out_dir, "frames.u8"), "rb").read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def parity() -> int:
    tiny = {
        "seq": 512,
        "ctx_len": 64,
        "heads": 4,
        "head_dim": 32,
        "layers": 2,
        "latent_channels": 16,
        "frames": 6,
        "latent_hw": 16,
        "out_hw": (64, 96),
        "seed": 20260917,
    }
    starter = load(STARTER, "starter_pipeline")
    oracle = load(ORACLE, "oracle_pipeline")
    starter.CFG = dict(tiny)
    oracle.CFG = dict(tiny)

    outs = {}
    for tag, mod in (("starter", starter), ("oracle", oracle)):
        d = tempfile.mkdtemp(prefix=f"parity_{tag}_")
        t0 = time.time()
        mod.run(d)
        outs[tag] = (d, time.time() - t0)
        print(f"  {tag:8s} ran in {outs[tag][1]:.2f}s -> {d}")

    shape = (tiny["frames"], tiny["out_hw"][0], tiny["out_hw"][1], 3)
    a = read_frames(outs["starter"][0], shape).astype(np.int16)
    b = read_frames(outs["oracle"][0], shape).astype(np.int16)

    diff = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    print(f"  mean|diff| = {diff.mean():.4f}   max|diff| = {diff.max()}   PSNR = {psnr:.1f} dB")

    # Gate thresholds the verifier will use.
    ok = diff.mean() <= 2.0 and diff.max() <= 12 and psnr >= 40.0
    print(f"  parity under verifier thresholds: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def budget() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device; budget check needs a GPU")
        return 2
    free, total = torch.cuda.mem_get_info(0)
    print(f"device: {torch.cuda.get_device_name(0)}  total={total / GIB:.1f} GiB  free={free / GIB:.1f} GiB")

    import torch.nn.functional as F

    starter = load(STARTER, "starter_for_budget")
    half = load(STARTER, "halffix_for_budget")
    # Fix ONLY the attention, leave the full-volume decode: must still OOM, which
    # is what substantiates the instruction's "fixing one of them is not enough".
    half._naive_sdpa = lambda q, k, v, scale: F.scaled_dot_product_attention(q, k, v, scale=scale)

    arms = [
        ("oracle", load(ORACLE, "oracle_for_budget"), 8.0, "must fit"),
        ("attn-fixed-only", half, 8.0, "must still OOM (decode hog untouched)"),
        ("starter", starter, 8.0, "must OOM"),
    ]

    rc = 0
    for tag, mod, budget_gib, expectation in arms:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget_gib * GIB / total), 0)
        d = tempfile.mkdtemp(prefix=f"budget_{tag}_")
        t0 = time.time()
        try:
            mod.run(d)
            peak = torch.cuda.max_memory_allocated(0) / GIB
            verdict = "PASS" if peak <= budget_gib else f"FAIL (> {budget_gib} GiB)"
            print(f"  {tag:16s} OK    wall={time.time() - t0:6.1f}s  peak_alloc={peak:6.2f} GiB  {verdict}")
            if tag != "oracle":
                print(f"  {'':16s}       UNEXPECTED -- {expectation}")
                rc = 1
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated(0) / GIB
            note = "expected" if tag != "oracle" else "UNEXPECTED -- oracle must fit"
            print(f"  {tag:16s} OOM   wall={time.time() - t0:6.1f}s  peak_alloc={peak:6.2f} GiB  ({note})")
            if tag == "oracle":
                rc = 1
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag:16s} ERROR {type(exc).__name__}: {exc}")
            rc = 1
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parity", "budget"])
    args = ap.parse_args()
    sys.exit(parity() if args.mode == "parity" else budget())
