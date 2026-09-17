"""Measure peak device memory of a pipeline under a hard budget.

The grader runs your /app/pipeline.py under exactly this mechanism (its copy is
the authoritative one; this is your local mirror of it). Use it to check a
candidate solution before you finish.

    python /app/vram_probe.py --pipeline /app/pipeline.py --budget-gib 8

What it does:

1. Spawns a child process with the PyTorch caching allocator capped at
   `budget_gib`, and with the device-memory *reporting* APIs patched so that
   `torch.cuda.get_device_properties(0).total_memory` and `mem_get_info()` agree
   with the budget. This is what a real 8 GiB card looks like to your code.
2. Samples NVML from this (parent) process for the duration of the child run.
   The printed peak is not something the child can influence.
3. Reports whether the child finished, and the peak it reached.

Exit code is 0 when the pipeline completed inside the budget, 1 when it hit the
budget (CUDA OOM), 2 when it failed for another reason.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import threading
import time

GIB = 1024**3

CHILD_SRC = textwrap.dedent(
    '''
    import json, os, sys, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, {probe_dir!r})

    budget = {budget}
    # --- same emulation the grader uses -------------------------------------
    import torch
    torch.cuda.init()
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / float(_total)), 0)

    _real_props = torch.cuda.get_device_properties
    def _props(device=None):
        p = _real_props(device)
        class _P: pass
        fake = _P()
        for a in dir(p):
            if not a.startswith("_"):
                try: setattr(fake, a, getattr(p, a))
                except Exception: pass
        fake.total_memory = budget
        return fake
    torch.cuda.get_device_properties = _props

    _real_mgi = torch.cuda.mem_get_info
    def _mgi(device=None):
        f, t = _real_mgi(device)
        return (max(0, budget - (t - f)), budget)
    torch.cuda.mem_get_info = _mgi
    # ------------------------------------------------------------------------

    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_pipeline", {pipeline!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t0 = time.time()
    try:
        mod.run({out_dir!r})
        status, err = "ok", None
    except torch.cuda.OutOfMemoryError as e:
        status, err = "oom", str(e).splitlines()[0]
    except Exception as e:
        status, err = "error", repr(e)
    print(json.dumps({{
        "status": status,
        "error": err,
        "wall_s": time.time() - t0,
        "torch_peak_allocated": int(torch.cuda.max_memory_allocated(0)),
        "torch_peak_reserved": int(torch.cuda.max_memory_reserved(0)),
    }}))
    '''
)


def _sample(out: dict, stop: threading.Event, interval: float = 0.05) -> None:
    try:
        import pynvml
    except Exception as exc:
        out["error"] = f"pynvml unavailable ({exc}); peak will be torch-only"
        return
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as exc:
        out["error"] = f"nvmlInit failed ({exc})"
        return
    while not stop.is_set():
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            used = sum(getattr(p, "usedGpuMemory", 0) or 0 for p in procs)
            out["peak_process"] = max(out.get("peak_process", 0), used)
        except Exception:
            pass
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out["peak_device"] = max(out.get("peak_device", 0), info.used)
        except Exception:
            pass
        time.sleep(interval)
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=os.path.join(here, "pipeline.py"))
    ap.add_argument("--budget-gib", type=float, default=8.0)
    ap.add_argument("--out-dir", default=os.path.join(here, "out"))
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    try:
        import torch

        if not torch.cuda.is_available():
            print("No CUDA device. This probe only measures GPU memory; run it on the GPU box.")
            return 2
    except Exception as exc:
        print(f"torch unavailable: {exc}")
        return 2

    code = CHILD_SRC.format(
        budget=int(args.budget_gib * GIB),
        pipeline=args.pipeline,
        out_dir=args.out_dir,
        probe_dir=here,
    )

    peak: dict = {}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample, args=(peak, stop), daemon=True)
    sampler.start()

    t0 = time.time()
    proc = subprocess.run([args.python, "-c", code], capture_output=True, text=True)
    wall = time.time() - t0
    stop.set()
    sampler.join(timeout=5)

    child: dict = {}
    for line in reversed((proc.stdout or "").strip().splitlines()):
        try:
            child = json.loads(line)
            break
        except Exception:
            continue

    peak_bytes = max(peak.get("peak_process", 0), peak.get("peak_device", 0))
    torch_peak = max(child.get("torch_peak_allocated", 0), child.get("torch_peak_reserved", 0))
    reported = max(peak_bytes, torch_peak)

    print(f"budget          : {args.budget_gib:.2f} GiB")
    print(f"status          : {child.get('status', 'no-report')}")
    if child.get("error"):
        print(f"error           : {child['error']}")
    print(f"wall            : {wall:.1f}s")
    print(f"peak (NVML)     : {peak_bytes / GIB:.2f} GiB   <- authoritative")
    print(f"peak (torch)    : {torch_peak / GIB:.2f} GiB")
    print(f"peak (reported) : {reported / GIB:.2f} GiB")
    print(f"headroom        : {(args.budget_gib * GIB - reported) / GIB:+.2f} GiB")
    if peak.get("error"):
        print(f"note            : {peak['error']}")
    if proc.returncode != 0 and not child:
        print("--- child stderr (tail) ---")
        print("\n".join((proc.stderr or "").strip().splitlines()[-15:]))

    if child.get("status") == "ok":
        return 0 if reported <= args.budget_gib * GIB else 1
    return 1 if child.get("status") == "oom" else 2


if __name__ == "__main__":
    sys.exit(main())
