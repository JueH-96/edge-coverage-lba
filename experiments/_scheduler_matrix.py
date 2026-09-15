"""Step 2 benchmark: >=30 independent trials per (target, scheduler) cell.

Follows the Klees et al. (CCS'18) protocol: at least 30 independent trials per
experimental cell, a randomized *block* design in which every condition sees the
same seed block (so the comparison is paired rather than merely independent),
right-censoring recorded explicitly instead of imputed, and a fixed execution
budget rather than a fixed wall-clock budget - the latter would confound the
schedule with the per-execution instrumentation overhead, which differs by design
between a D0 and a 3D condition.

Compute note
------------
The fuzzing loop is sequential, branch-dominated, and mutates Python dict/set
state per execution on inputs of a few dozen bytes. Per the ``optimize-for-gpu``
skill's suitability criteria, that is the profile it explicitly says to keep on
CPU: no substantial independent work inside the hot path, a working set far too
small to amortize a host-device transfer, and control flow that is data-dependent
by construction. The parallelism here is at the *trial* level and is
embarrassingly so, which is why the work is distributed across CPU worker
processes. The GPU is probed at runtime and the finding is recorded in the
results rather than left implicit.

Usage
-----
``uv run python workflow/02_run_step2_experiments.py [--trials 30] [--execs 50000]``
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

# --- artifact-evaluation path bootstrap (replaces the session-root anchor) --- #
import sys as _ae_sys  # noqa: E402
from pathlib import Path as _AEPath  # noqa: E402

AE_ROOT = _AEPath(__file__).resolve().parent.parent
if str(AE_ROOT) not in _ae_sys.path:
    _ae_sys.path.insert(0, str(AE_ROOT))
import ae_paths as _ae_paths  # noqa: E402,F401  (registers src/ and targets/)

SESSION = AE_ROOT
# --- end bootstrap --------------------------------------------------------- #
sys.path.insert(0, str(AE_ROOT / "src"))

from energy_scheduler import DEPTH_LADDER, ENERGY_BASE, ENERGY_MAX, ENERGY_MIN  # noqa: E402
from step2_harness import STEP2_CONFIGS, step2_worker, trial_seed  # noqa: E402
from trigram_targets import build_trigram_lib, get_trigram_targets  # noqa: E402

#: Master seed for the whole Step 2 experiment. Every trial seed is derived from
#: it by :func:`trial_seed`, so the entire matrix is reproducible from this one
#: integer.
MASTER_SEED = 20260811

OUT = SESSION / "data" / "step2_trials.json"


def gpu_probe() -> Dict[str, Any]:
    """Detect NVIDIA GPUs and record the applicability finding.

    Returns
    -------
    dict
        Availability, device list, and the reason the fuzzing loop does not use
        the device.
    """
    info: Dict[str, Any] = {
        "cuda_available": False,
        "devices": [],
        "used_for_fuzzing_loop": False,
        "rationale": (
            "The fuzzing loop is sequential and branch-dominated: each execution "
            "mutates a few dozen input bytes, walks a data-dependent control "
            "path, and updates hash-map tracker state. It exposes no wide "
            "independent work, and its working set is far too small to amortize "
            "a host-device transfer, so it matches the CPU-retention criteria in "
            "the optimize-for-gpu guidance. Parallelism is exploited at the "
            "trial level across CPU worker processes instead."
        ),
    }
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            info["cuda_available"] = True
            info["devices"] = [ln.strip() for ln in proc.stdout.strip().splitlines()]
    except Exception as exc:  # pragma: no cover - probe is best effort
        info["probe_error"] = str(exc)
    return info


def main() -> int:
    """Run the full matrix and write the raw trial records."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=30, help="trials per cell (>=30)")
    ap.add_argument("--execs", type=int, default=50000, help="execution budget")
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--curve-every", type=int, default=1000)
    args = ap.parse_args()

    if args.trials < 30:
        raise SystemExit("the Klees et al. protocol requires at least 30 trials/cell")

    build_trigram_lib()
    targets = [t.NAME for t in get_trigram_targets()]
    configs = [c.name for c in STEP2_CONFIGS]

    jobs: List[Dict[str, Any]] = []
    for tgt in targets:
        for cfg in configs:
            for i in range(args.trials):
                jobs.append(
                    {
                        "target": tgt,
                        "config": cfg,
                        # Blocked seed: trial i of EVERY condition on a target
                        # shares one seed, which makes the design paired and
                        # licenses the paired tests in the analysis.
                        "seed": trial_seed(MASTER_SEED, tgt, i),
                        "max_execs": args.execs,
                        "curve_every": args.curve_every,
                    }
                )

    print(
        f"Step 2 matrix: {len(targets)} targets x {len(configs)} configs x "
        f"{args.trials} trials = {len(jobs)} trials, budget {args.execs} execs",
        flush=True,
    )
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(step2_worker, jobs, chunksize=1):
            results.append(r)
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                print(
                    f"  {done}/{len(jobs)} ({100 * done / len(jobs):.1f}%) "
                    f"elapsed={el:.0f}s eta={el / done * (len(jobs) - done):.0f}s",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0
    print(f"matrix complete in {elapsed:.0f}s", flush=True)

    payload = {
        "master_seed": MASTER_SEED,
        "n_trials_per_cell": args.trials,
        "max_execs": args.execs,
        "curve_every": args.curve_every,
        "targets": targets,
        "configs": [
            {
                "name": c.name,
                "dims": list(c.dims),
                "schedule": c.schedule,
                "context_depth": c.context_depth,
                "dynamic_depth": c.dynamic_depth,
                "use_feedback": c.use_feedback,
                "description": c.description,
            }
            for c in STEP2_CONFIGS
        ],
        "energy_bounds": {
            "min": ENERGY_MIN,
            "max": ENERGY_MAX,
            "base": ENERGY_BASE,
            "depth_ladder": list(DEPTH_LADDER),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "n_workers": args.workers,
            "wall_seconds": elapsed,
            "gpu": gpu_probe(),
        },
        "trials": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, allow_nan=False))
    print(f"raw trials -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
