"""Corpus-capped ablation on T4: isolating dilution from discrimination.

The order sweep shows that feedback order 3 beats order 2 on T4 and that orders
4, 6 and 8 are much worse than either.  Two mechanisms are confounded in that
observation:

``dilution``
    finer feedback admits more corpus entries, and a round-robin scheduler
    divides a fixed execution budget over all of them, so each entry receives
    fewer mutations;

``admission quality``
    finer feedback may admit entries that are individually less useful,
    independently of how many there are.

This script holds the *feedback function* fixed and caps the *corpus capacity*.
When the corpus is full, a newly interesting input replaces a uniformly random
non-seed entry (reservoir-style replacement), so the fuzzer keeps reacting to
new coverage but the scheduler's sweep length is pinned.  If dilution is the
causal mechanism, capping the corpus of a high-order arm at the corpus size of
the edge-coverage baseline should recover much of the lost performance.

Everything else -- targets, mutation operators, seed corpus, budget, blocked
trial seeds -- is identical to the main order sweep, so capped and uncapped arms
are matched block for block.

Usage::

    python corpus_cap_ablation.py [--trials 30] [--budget 400000] [--workers 24]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# --- artifact-evaluation path bootstrap (replaces the session-root anchor) --- #
import sys as _ae_sys  # noqa: E402
from pathlib import Path as _AEPath  # noqa: E402

AE_ROOT = _AEPath(__file__).resolve().parent.parent
if str(AE_ROOT) not in _ae_sys.path:
    _ae_sys.path.insert(0, str(AE_ROOT))
import ae_paths as _ae_paths  # noqa: E402,F401  (registers src/ and targets/)

SESSION_DIR = AE_ROOT
# --- end bootstrap --------------------------------------------------------- #
sys.path.insert(0, str(AE_ROOT / "src"))

import numpy as np  # noqa: E402

from fuzz_harness import (  # noqa: E402
    CONFIGS,
    CONFIG_PARAMS,
    Mutator,
    SEED_INPUT,
    SEED_INPUT_BY_TARGET,
    TrialRecord,
    trial_seed,
)
from guidance_engine import MultiDimGuidanceEngine  # noqa: E402
from stats_paired import (  # noqa: E402
    mcnemar_exact,
    paired_prob_superiority,
    paired_win_loss_tie,
)
from stats_utils import benjamini_hochberg, clopper_pearson  # noqa: E402
from test_targets import get_targets  # noqa: E402

MASTER_SEED = 20260804
TARGET = "T4_trigram"
OUT = AE_ROOT / "results" / "corpus_cap_ablation.json"

#: Capacity levels.  ``2170`` is the median corpus the edge-coverage baseline
#: reaches on T4; ``18864`` is the median corpus of the order-3 arm.  Capping a
#: high-order arm at one of these makes its scheduler sweep length equal to that
#: of the arm it is being compared against.
CAP_EDGE = 2170
CAP_NGRAM3 = 18864


def run_capped_trial(
    config_name: str,
    seed: int,
    max_execs: int,
    corpus_cap: int | None,
    curve_every: int = 10000,
    map_bits: int = 16,
    context_depth: int = 16,
    kgram: int = 3,
    ngram: int = 3,
) -> TrialRecord:
    """One trial of the T4 campaign with a bounded corpus.

    Identical to ``fuzz_harness.run_trial`` under the round-robin schedule
    except that the corpus never exceeds ``corpus_cap`` entries: once full, an
    admitted input overwrites a uniformly random entry other than the seed.
    ``corpus_cap=None`` reproduces the unbounded behaviour exactly.
    """
    targets = {t.NAME: t for t in get_targets()}
    target = targets[TARGET]
    dims = CONFIGS[config_name]
    over = CONFIG_PARAMS.get(config_name, {})
    context_depth = int(over.get("context_depth", context_depth))
    ngram = int(over.get("ngram", ngram))

    rng = random.Random(seed)
    mut = Mutator(rng)
    engine = MultiDimGuidanceEngine(
        dims=dims,
        map_bits=map_bits,
        context_depth=context_depth,
        kgram=kgram,
        ngram=ngram,
        track_exact=True,
    )

    seed_input = SEED_INPUT_BY_TARGET.get(TARGET, SEED_INPUT)
    rec = TrialRecord(target=TARGET, config=config_name, seed=seed)
    corpus: List[bytes] = [seed_input]
    n_admitted = 0
    n_evicted = 0

    engine.reset()
    target.run(seed_input, engine)
    engine.evaluate(commit=True)

    cursor = 0
    execs = 0
    t0 = time.perf_counter()
    while execs < max_execs:
        parent = corpus[cursor % len(corpus)]
        cursor += 1
        data = mut.mutate(parent, splice_pool=corpus)

        engine.reset()
        res = target.run(data, engine)
        fb = engine.evaluate(commit=True)
        execs += 1

        if res.milestone > rec.max_milestone:
            for lvl in range(rec.max_milestone + 1, res.milestone + 1):
                rec.milestone_first_exec[lvl] = execs
            rec.max_milestone = res.milestone

        if res.bug:
            rec.n_bugs += 1
            if rec.first_bug_exec is None:
                rec.first_bug_exec = execs
            break

        if fb.is_interesting:
            rec.n_interesting += 1
            n_admitted += 1
            if corpus_cap is not None and len(corpus) >= corpus_cap:
                victim = rng.randrange(1, len(corpus))
                corpus[victim] = data
                n_evicted += 1
            else:
                corpus.append(data)

        if execs % curve_every == 0:
            st = engine.stats()
            rec.curve.append(
                (execs, int(st["total_exact_elements"]), len(corpus), rec.max_milestone)
            )

    rec.execs = execs
    rec.wall_seconds = time.perf_counter() - t0
    rec.corpus_size = len(corpus)
    st = engine.stats()
    rec.engine_stats = {
        "total_exact_elements": int(st["total_exact_elements"]),
        "n_admitted": n_admitted,
        "n_evicted": n_evicted,
    }
    return rec


def worker(job: Dict[str, Any]) -> Dict[str, Any]:
    job = dict(job)
    tag = job.pop("tag")
    arm = job.pop("arm")
    block = job.pop("block")
    rec = run_capped_trial(**job)
    return {
        "arm": arm,
        "config": rec.config,
        "cap": job["corpus_cap"],
        "block": block,
        "tag": tag,
        "seed": rec.seed,
        "execs": rec.execs,
        "wall_seconds": rec.wall_seconds,
        "corpus_size": rec.corpus_size,
        "n_interesting": rec.n_interesting,
        "first_bug_exec": rec.first_bug_exec,
        "max_milestone": rec.max_milestone,
        "engine_stats": rec.engine_stats,
    }


#: The ablation matrix.  ``edge_only`` and the uncapped high-order arms are
#: re-run here rather than copied from the main sweep so that every number in
#: the ablation comes from one campaign under one code path.
ARMS: Tuple[Tuple[str, str, int | None], ...] = (
    ("edge_only", "edge_only", None),
    ("ngram3", "ngram3", None),
    ("ngram4", "ngram4", None),
    ("ngram6", "ngram6", None),
    ("ngram8", "ngram8", None),
    ("ngram3_cap_edge", "ngram3", CAP_EDGE),
    ("ngram4_cap_edge", "ngram4", CAP_EDGE),
    ("ngram6_cap_edge", "ngram6", CAP_EDGE),
    ("ngram8_cap_edge", "ngram8", CAP_EDGE),
    ("ngram4_cap_n3", "ngram4", CAP_NGRAM3),
    ("ngram8_cap_n3", "ngram8", CAP_NGRAM3),
)


def arm_vectors(recs: Sequence[Dict[str, Any]], arm: str, budget: int):
    rows = sorted([r for r in recs if r["arm"] == arm], key=lambda r: r["block"])
    t = np.array(
        [r["first_bug_exec"] if r["first_bug_exec"] else r["execs"] for r in rows],
        dtype=float,
    )
    e = np.array([1 if r["first_bug_exec"] else 0 for r in rows], dtype=int)
    return {
        "t": t,
        "e": e,
        "interesting": np.array([r["n_interesting"] for r in rows], dtype=float),
        "execs": np.array([r["execs"] for r in rows], dtype=float),
        "corpus": np.array([r["corpus_size"] for r in rows], dtype=float),
        "ms": np.array([r["max_milestone"] for r in rows], dtype=float),
        "elements": np.array(
            [r["engine_stats"]["total_exact_elements"] for r in rows], dtype=float
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--budget", type=int, default=400000)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    jobs = []
    for arm, cfg, cap in ARMS:
        for i in range(args.trials):
            jobs.append(
                {
                    "arm": arm,
                    "config_name": cfg,
                    "corpus_cap": cap,
                    "seed": trial_seed(MASTER_SEED, TARGET, i),
                    "max_execs": args.budget,
                    "block": i,
                    "tag": f"{arm}#{i}",
                }
            )

    t0 = time.perf_counter()
    with mp.Pool(processes=args.workers) as pool:
        recs = pool.map(worker, jobs, chunksize=1)
    wall = time.perf_counter() - t0

    per_arm: Dict[str, Any] = {}
    for arm, cfg, cap in ARMS:
        v = arm_vectors(recs, arm, args.budget)
        succ = int(v["e"].sum())
        hits = v["t"][v["e"] == 1]
        per_arm[arm] = {
            "config": cfg,
            "cap": cap,
            "n_trials": args.trials,
            "n_success": succ,
            "success_rate": succ / args.trials,
            "success_ci95": list(clopper_pearson(succ, args.trials)),
            "median_execs_to_bug": float(np.median(hits)) if succ else None,
            "median_corpus": float(np.median(v["corpus"])),
            "median_elements": float(np.median(v["elements"])),
            "median_max_milestone": float(np.median(v["ms"])),
            "median_n_admitted": float(np.median(v["interesting"])),
            "admission_rate": float(np.median(v["interesting"] / v["execs"])),
            "expected_residence_execs": (
                float(np.median(v["corpus"] * v["execs"] / np.maximum(v["interesting"], 1)))
                if cap is not None
                else None
            ),
        }

    # Paired contrasts: each capped arm against its own uncapped twin, and each
    # capped arm against the edge-coverage control.
    contrasts = []
    pairs = [
        ("ngram3_cap_edge", "ngram3"),
        ("ngram4_cap_edge", "ngram4"),
        ("ngram6_cap_edge", "ngram6"),
        ("ngram8_cap_edge", "ngram8"),
        ("ngram4_cap_n3", "ngram4"),
        ("ngram8_cap_n3", "ngram8"),
        ("ngram4_cap_edge", "edge_only"),
        ("ngram8_cap_edge", "edge_only"),
        ("ngram3", "edge_only"),
        ("ngram4", "edge_only"),
    ]
    for a, b in pairs:
        va, vb = arm_vectors(recs, a, args.budget), arm_vectors(recs, b, args.budget)
        wlt = paired_win_loss_tie(va["t"], va["e"], vb["t"], vb["e"])
        mcn = mcnemar_exact(va["e"].tolist(), vb["e"].tolist())
        psup = paired_prob_superiority(va["t"], va["e"], vb["t"], vb["e"])
        contrasts.append(
            {
                "arm": a,
                "vs": b,
                "succ_arm": int(va["e"].sum()),
                "succ_vs": int(vb["e"].sum()),
                "win_loss_tie": wlt,
                "mcnemar": mcn,
                "prob_superiority": psup,
            }
        )
    pvals = [c["mcnemar"]["p_value"] for c in contrasts]
    qvals = benjamini_hochberg(pvals)["q_values"]
    for c, q in zip(contrasts, qvals):
        c["mcnemar_q"] = float(q)

    out = {
        "title": "corpus-capped ablation on T4 (dilution vs admission quality)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "master_seed": MASTER_SEED,
            "workers": args.workers,
        },
        "design": {
            "target": TARGET,
            "budget_execs": args.budget,
            "n_trials_per_arm": args.trials,
            "schedule": "round_robin",
            "cap_policy": "reservoir replacement of a uniformly random non-seed entry",
            "cap_levels": {"edge": CAP_EDGE, "ngram3": CAP_NGRAM3},
            "blocking": "trial i uses the same derived seed in every arm",
        },
        "per_arm": per_arm,
        "contrasts": contrasts,
        "multiple_testing_correction": {
            "method": "Benjamini-Hochberg FDR",
            "family": "all paired contrasts in this ablation",
            "n_tests": len(contrasts),
        },
        "wall_seconds": wall,
    }
    OUT.write_text(json.dumps(out, indent=1))
    (HERE_TRIALS := AE_ROOT / "data" / "corpus_cap_trials.json").write_text(json.dumps(recs))
    print(json.dumps({k: v for k, v in per_arm.items()}, indent=1))
    for c in contrasts:
        print(
            f"{c['arm']:>18s} vs {c['vs']:<12s} "
            f"{c['succ_arm']:>2d}/{args.trials} vs {c['succ_vs']:>2d}/{args.trials} "
            f"W/L={c['win_loss_tie']['wins_a']}/{c['win_loss_tie']['wins_b']} "
            f"q={c['mcnemar_q']:.4f}"
        )
    print(f"wall {wall:.1f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
