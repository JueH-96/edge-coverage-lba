#!/usr/bin/env python3
"""T4 n-gram order sweep: how far up the order ladder does the blind spot go?

This is the campaign behind the paper's order-sweep figure and the discrimination-
by-order table.  The T4 target is parameterised by the n-gram order its bug depends
on; guidance configurations track prefix contexts of order NGRAM = 2..8, plus the
``edge_only`` (AFL) control and a ``blind_random`` floor.  The prediction under test
is the one that follows from the AFL-bigram characterisation: edge coverage should
be blind to structure of order >= 3, and guidance of order k should discriminate the
target exactly when k >= the order the bug is built on.

Both energy schedules used in the paper are run (``score`` and ``round_robin``), so
the order effect can be separated from the scheduling effect.

Cost
----
Full configuration (30 trials x 9 configs x 2 schedules x 150k execs) takes on the
order of an hour on ~12 workers.  ``--pilot`` runs a 5-trial / 40k-exec version in a
few minutes; it reproduces the qualitative ordering but not the paper's confidence
intervals.

The bundled ``data/t4_order_sweep.json`` holds the trial records from the run
reported in the paper, so ``run_ae.sh --quick`` never needs to execute this.

Usage
-----
    python experiments/run_order_sweep.py --pilot
    python experiments/run_order_sweep.py --trials 30 --budget 150000 --workers 12

Outputs
-------
``data/t4_order_sweep.json``   raw per-trial records (overwrites the bundled copy)
``results/step2_revision.json``  the analysed order sweep (E2 section)
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
    """Delegate to the revision-experiment engine, restricted to the E2 sweep."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true",
                    help="fast 5-trial / 40k-exec configuration")
    ap.add_argument("--trials", type=int, default=30, help="trials per cell")
    ap.add_argument("--budget", type=int, default=150_000, help="execs per trial")
    ap.add_argument("--workers", type=int, default=12, help="parallel worker processes")
    args = ap.parse_args()

    sys.path.insert(0, str(AE_ROOT / "experiments"))
    import _revision_experiments as rev

    # The engine parses its own argv; hand it exactly the E2 selection.
    argv = ["run_order_sweep", "--only", "e2",
            "--trials", str(args.trials),
            "--budget", str(args.budget),
            "--workers", str(args.workers)]
    if args.pilot:
        argv.append("--pilot")
    old_argv, sys.argv = sys.argv, argv
    try:
        print(f"[order-sweep] NGRAM 2-8 sweep on T4, "
              f"{'PILOT' if args.pilot else 'FULL'} configuration", flush=True)
        return rev.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    sys.exit(main())
