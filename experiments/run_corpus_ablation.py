#!/usr/bin/env python3
"""Fixed-corpus-capacity dilution ablation (K = 1000).

The corpus-dilution hypothesis says that a more sensitive feedback signal is not
free: it admits more inputs into the corpus, and under a *finite* corpus budget the
extra entries crowd out the productive ones.  The confound in the main campaign is
that sensitivity and corpus size move together, so a sensitivity effect and a
capacity effect cannot be told apart.

This ablation removes the confound by holding corpus capacity fixed at K = 1000
across all arms.  If the higher-order guidance still loses ground at fixed capacity,
dilution is doing the damage rather than the extra bookkeeping cost.

Cost
----
Full configuration (30 trials x 400k execs) takes tens of minutes on ~24 workers.
Lower ``--trials``/``--budget`` for a smoke run, at the price of statistical power.

The bundled ``data/corpus_cap_trials.json`` holds the trial records from the run
reported in the paper, and ``data/precomputed/corpus_cap_ablation.json`` the analysed
form, so ``run_ae.sh --quick`` never needs to execute this.

Usage
-----
    python experiments/run_corpus_ablation.py --trials 30 --budget 400000 --workers 24

Outputs
-------
``data/corpus_cap_trials.json``      raw per-trial records
``results/corpus_cap_ablation.json``  per-arm analysis at fixed capacity
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
    """Delegate to the ported fixed-capacity ablation engine."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=30, help="trials per arm")
    ap.add_argument("--budget", type=int, default=400_000, help="execs per trial")
    ap.add_argument("--workers", type=int, default=24, help="parallel worker processes")
    args = ap.parse_args()

    sys.path.insert(0, str(AE_ROOT / "experiments"))
    import _corpus_cap as cap

    argv = ["run_corpus_ablation",
            "--trials", str(args.trials),
            "--budget", str(args.budget),
            "--workers", str(args.workers)]
    old_argv, sys.argv = sys.argv, argv
    try:
        print(f"[cap-ablation] fixed corpus capacity K=1000, {args.trials} trials/arm, "
              f"{args.budget} execs/trial", flush=True)
        return cap.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    sys.exit(main())
