#!/usr/bin/env python3
"""The full energy-schedule campaign matrix: 3 targets x 9 arms x 30 trials.

This is the paper's main fuzzing experiment.  Every arm runs the *same* mutator over
the *same* blocked seed sequence and differs only in two factors:

* the feedback oracle -- ``blind_random`` (no feedback), ``afl_d0_*`` (edge coverage
  only), or ``d3d_*`` (the 3D multidimensional guidance), and
* the energy schedule -- round-robin, AFLFast-style exponential, static novelty
  weights, adaptive novelty, or dynamic context-depth escalation.

Because seeds are blocked across arms, arm-to-arm differences are attributable to
those two factors rather than to seed luck.  The design follows Klees et al. (CCS'18):
>= 30 independent trials per cell, right-censoring recorded explicitly, and no
cherry-picked run lengths.

Cost
----
Full configuration (810 trials x 50k execs) took ~334 s wall on 26 workers for the
run reported in the paper.  Scale ``--workers`` to the machine; the campaign is
embarrassingly parallel across trials.  This is CPU-bound, branch-heavy interpreted
work driving a ctypes-loaded C library -- there is no GPU-amenable kernel here, so
the runner parallelises across processes rather than devices.

The bundled ``data/step2_trials.json`` holds the trial records from the paper run,
so ``run_ae.sh --quick`` never needs to execute this.

Usage
-----
    python experiments/run_scheduler_matrix.py --trials 30 --execs 50000 --workers 26
    python experiments/run_scheduler_matrix.py --trials 5 --execs 10000 --workers 8

Outputs
-------
``data/step2_trials.json``  raw per-trial records (overwrites the bundled copy)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

AE_ROOT = Path(__file__).resolve().parent.parent
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))

import ae_paths  # noqa: E402,F401


def main() -> int:
    """Delegate to the ported campaign runner."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=30,
                    help="trials per cell (>=30 for the paper's protocol)")
    ap.add_argument("--execs", type=int, default=50_000, help="execution budget/trial")
    ap.add_argument("--workers", type=int, default=26, help="parallel worker processes")
    ap.add_argument("--curve-every", type=int, default=1000,
                    help="progress-curve sampling interval in execs")
    args = ap.parse_args()

    if args.trials < 30:
        print(f"[matrix] NOTE: {args.trials} trials/cell is below the 30 required by "
              "the paper's protocol; results will be under-powered relative to the "
              "reported confidence intervals.", flush=True)

    sys.path.insert(0, str(AE_ROOT / "experiments"))
    import _scheduler_matrix as matrix

    argv = ["run_scheduler_matrix",
            "--trials", str(args.trials),
            "--execs", str(args.execs),
            "--workers", str(args.workers),
            "--curve-every", str(args.curve_every)]
    old_argv, sys.argv = sys.argv, argv
    try:
        print(f"[matrix] 3 targets x 9 arms x {args.trials} trials "
              f"= {3 * 9 * args.trials} trials at {args.execs} execs each", flush=True)
        return matrix.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    sys.exit(main())
