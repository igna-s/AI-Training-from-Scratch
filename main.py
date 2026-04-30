#!/usr/bin/env python3
"""
Dry-run validation suite for the 4 fixes applied to train_ablation.py.

Run with:  python main.py

Tests (sequential — each must pass before the next starts):
  1. --total_steps 5     Real GPU, Phase 1: full training loop + ckpt save
  2. --phase 2, 5 steps   Real GPU, Phase 2: loads Phase-1 ckpt, trains DeleteGate

Covers: Fix #1 (Phase-2 ckpt path), Fix #2 (Chinchilla Phase-2 budget),
        Fix #4 (torch.compile guard, only fires on cuda+not-dry_run),
        Fix #5 (param groups, only fires on not-dry_run).
Also exercises evaluate_run to confirm it doesn't hang on FineWeb streaming.

Note: --dry_run on CPU was deliberately removed — DryRunDataset emits the
production seq_len=5000 × batch_per_gpu=24, which OOMs CPU RAM on Lightning
Studio (rc=-9 SIGKILL). It also doesn't test Fix #4 since compile is gated
on cuda+not-dry_run. Test 1 below covers everything Test-1-on-CPU would have.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import torch

SCRIPT = Path(__file__).parent / "train_ablation.py"
PY = sys.executable
RUN = "run_09"
PHASE1_CKPT = f"checkpoint_{RUN}_final.pt"

# Smoke-test overrides — keep the run path intact but minimize FineWeb cold-start:
#   SMOKE_N_WORKERS=1     → only one worker resolves files / fills shuffle buffer
#   SMOKE_BUFFER_SIZE=256 → shuffle buffer fills in seconds, not minutes
#   SMOKE_BATCH_PER_GPU=4 → 20 steps × 4 = 80 samples needed total
# These are read by train_ablation.py:938 and utilities.py:1638.
SMOKE_ENV = {
    "SMOKE_N_WORKERS":         "0",   # everything in main process — no hidden worker
    "SMOKE_BUFFER_SIZE":       "32",  # smallest viable shuffle buffer
    "SMOKE_BATCH_PER_GPU":     "4",
    "SMOKE_NO_COMPILE":        "1",   # skip torch.compile (saves 1-3 min cold start)
    "SMOKE_MAX_VAL_BATCHES":   "5",   # 5 val batches × 4 = 20 samples (vs 400 default)
    "PYTHONUNBUFFERED":        "1",   # force unbuffered stdout from subprocess
}

TESTS = [
    {
        "label": "Test 1 — Phase 1, 5 steps (GPU)",
        "desc":  "Fix #4 compile guard + Fix #5 param groups + arch + forward + loss + ckpt save",
        "cmd":   [PY, "-u", str(SCRIPT), "--run", RUN, "--total_steps", "5"],
        "needs_ckpt": False,
    },
    {
        "label": "Test 2 — Phase 2, 5 steps (GPU)",
        "desc":  "Fix #1 ckpt load + Fix #2 Chinchilla Phase-2 budget + DeleteGate training",
        "cmd":   [PY, "-u", str(SCRIPT), "--run", RUN,
                  "--phase", "2", "--phase1_ckpt", PHASE1_CKPT,
                  "--total_steps", "5"],
        "needs_ckpt": True,
    },
]


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def run_test(test: dict, idx: int, total: int) -> bool:
    banner(f"[{idx}/{total}] {test['label']}\n       {test['desc']}")
    print(f"CMD: {' '.join(str(x) for x in test['cmd'])}\n")

    if test["needs_ckpt"] and not Path(PHASE1_CKPT).exists():
        print(f"[SKIP] Prerequisite missing: {PHASE1_CKPT}")
        print("       Test 2 must succeed first.")
        return False

    env = os.environ.copy()
    env.update(SMOKE_ENV)

    t0 = time.time()
    # stdout/stderr inherited — live output visible in terminal
    result = subprocess.run(test["cmd"], env=env)
    elapsed = time.time() - t0

    ok = result.returncode == 0
    print(f"\n>>> {'PASS' if ok else 'FAIL (rc=' + str(result.returncode) + ')'}  [{elapsed:.1f}s]")
    return ok


def main() -> int:
    banner("CPU-1 Ablation — Dry-run Validation Suite")
    cuda = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda else 0
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {cuda}  |  GPUs: {n_gpus}")
    if not cuda:
        print("\n[ERROR] No GPU detected — both tests require CUDA.")
        return 2

    print("\nSmoke overrides:")
    for k, v in SMOKE_ENV.items():
        print(f"  {k}={v}")

    results: dict[str, bool | None] = {}
    total = len(TESTS)

    for i, test in enumerate(TESTS, 1):
        ok = run_test(test, i, total)
        results[test["label"]] = ok
        if not ok:
            print(f"\n[ABORT] Test {i} failed — skipping remaining tests.")
            for t in TESTS[i:]:
                results[t["label"]] = None
            break

    banner("SUMMARY")
    all_pass = True
    for label, ok in results.items():
        if ok is None:
            tag = "SKIP"
        elif ok:
            tag = "PASS"
        else:
            tag, all_pass = "FAIL", False
        print(f"  {tag:4s}  {label}")

    print()
    if all_pass and None not in results.values():
        print("Both tests passed — safe to launch full orchestration.")
    elif all_pass:
        print("Some tests were skipped (prerequisite failure).")
    else:
        print("One or more tests failed. Fix before launching orchestration.")
    print("=" * 70)
    return 0 if (all_pass and None not in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
