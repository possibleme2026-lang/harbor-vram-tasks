"""Run the grader's tests without pytest. Task-author tooling only.

Local fast path: it imports tests/test_outputs.py and calls the same test
functions in the same order as pytest would, building the session fixtures by
hand. The authoritative runner is still pytest inside the verifier image
(tests/test.sh) -- this driver exists so a dev box can iterate without building
that image, and it is only trustworthy if the call list below stays in sync with
the `def test_*` order in tests/test_outputs.py. There are 8 of them; when you
add or remove one, update this list too or the local partial score will silently
disagree with CI's.

    python dev/run_grader.py --pipeline <path to a candidate pipeline.py>

Expectations:
    oracle_pipeline.py                       -> all 8 tests pass
    environment/app/pipeline.py (starter)    -> the memory/run tests fail
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", "tasks", "wan-vram-budget"))
TESTS = os.path.join(TASK, "tests")
STARTER = os.path.join(TASK, "environment", "app", "pipeline.py")
ORACLE = os.path.join(TESTS, "reference", "oracle_pipeline.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=ORACLE)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="grader_agent_")
    os.environ["TB_AGENT_PIPELINE"] = args.pipeline
    os.environ["TB_AGENT_OUT_DIR"] = out_dir
    sys.path.insert(0, TESTS)

    import test_outputs as T  # noqa: N812

    print(f"agent pipeline : {T.AGENT_PIPELINE}")
    print(f"agent out dir  : {T.AGENT_OUT_DIR}")
    print(f"reference      : {T.ORACLE}")

    results: list[tuple[str, str]] = []

    def record(name: str, fn, *a) -> None:
        try:
            fn(*a)
            results.append((name, "PASS"))
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            kind = type(exc).__name__
            results.append((name, "SKIP" if kind == "Skipped" else "FAIL"))
            print(f"  {results[-1][1]}  {name}: {kind}: {str(exc).splitlines()[0][:200]}")
            if os.environ.get("TB_VERBOSE"):
                traceback.print_exc()

    print("\n--- building fixtures ---")
    ref_dir = tempfile.mkdtemp(prefix="grader_ref_frames_")
    reference = T._run_under_budget(T.ORACLE, ref_dir, "reference")
    reference["frames_path"] = os.path.join(ref_dir, "frames.u8")

    agent = T._run_under_budget(T.AGENT_PIPELINE, out_dir, "agent")
    agent["frames_path"] = T.AGENT_FRAMES

    print("\n--- tests ---")
    record("test_gpu_is_visible", T.test_gpu_is_visible)
    record("test_reference_fits_budget", T.test_reference_fits_budget, reference)
    record("test_reference_is_nondegenerate", T.test_reference_is_nondegenerate, reference)
    record("test_pipeline_config_unmodified", T.test_pipeline_config_unmodified)
    record("test_pipeline_runs_within_budget", T.test_pipeline_runs_within_budget, agent)
    record("test_peak_memory_within_budget", T.test_peak_memory_within_budget, agent)
    record("test_output_files_present", T.test_output_files_present, agent)
    record("test_frames_match_reference", T.test_frames_match_reference, agent, reference)

    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\npartial = {passed}/{total} = {passed / total:.4f}   binary reward = {1.0 if passed == total else 0.0:.1f}")

    print("\n--- agent report ---")
    for key in ("status", "error", "wall_s", "measured_peak_bytes",
                "torch_peak_allocated_bytes", "torch_peak_reserved_bytes"):
        if key in agent:
            val = agent[key]
            if key.endswith("_bytes"):
                val = f"{val / 1024 ** 3:.2f} GiB"
            print(f"  {key:28s} {val}")
    if agent.get("nvml", {}).get("error"):
        print(f"  nvml note: {agent['nvml']['error']}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
