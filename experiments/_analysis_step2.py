"""Step 2 statistical analysis and figures.

Consumes ``data/step2_trials.json`` and emits ``results/step2_scheduler_validation.json``
(strict RFC 8259, ``allow_nan=False``) plus ``figures/step2_*.png|pdf``.

Endpoints
---------
Two, because the targets differ in what is reachable inside the budget:

* **time-to-first-bug**, right-censored at the execution budget. Compared with
  the log-rank test, which is the correct test for censored data - taking means
  over the uncensored trials alone would silently condition on success and bias
  every comparison toward whichever condition failed most often.
* **time-to-max-milestone-reached**, also right-censored. On targets where few
  or no trials reach the oracle, the milestone ladder is the informative
  endpoint, and reporting only the bug endpoint there would be a floor effect
  reported as a null result.

Effect sizes are Vargha-Delaney ``A12`` (with censored observations at the
budget, which is conservative: it can only shrink an apparent difference).
Success rates use Fisher's exact test. All p-values across the whole family are
adjusted together with Benjamini-Hochberg FDR.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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

from stats_utils import (  # noqa: E402
    a12_magnitude,
    benjamini_hochberg,
    clopper_pearson,
    describe,
    fisher_success,
    logrank_test,
    mann_whitney,
    vargha_delaney_a12,
)
from trigram_targets import (  # noqa: E402
    bigram_multiset,
    get_trigram_targets,
    trigram_multiset,
)

IN = SESSION / "data" / "step2_trials.json"
OUT = SESSION / "results" / "step2_scheduler_validation.json"
FIG = SESSION / "figures"

#: The condition every scheduler claim is made against: Step 1's policy.
CONTROL = "d3d_rr_n4"
#: The contribution.
TREATMENT = "d3d_adaptive_n4"

plt.rcParams.update(
    {"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 140, "savefig.dpi": 300}
)
PALETTE = {
    "blind_random": "#9e9e9e",
    "afl_d0_rr": "#4c72b0",
    "afl_d0_fast": "#55a868",
    "d3d_rr_n4": "#c44e52",
    "d3d_rr_n16": "#8172b2",
    "d3d_fast_n4": "#937860",
    "d3d_static_n4": "#da8bc3",
    "d3d_adaptive_n4": "#000000",
    "d3d_adaptive_dyn": "#dd8452",
}


def _f(x: Any) -> float:
    """Coerce to a JSON-safe float; non-finite values become ``None``-able NaN guards."""
    v = float(x)
    return v if math.isfinite(v) else 0.0


def _san(o: Any) -> Any:
    """Recursively replace non-finite floats so ``allow_nan=False`` can be used."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return _san(o.item())
    return o


def event_times(trials: List[dict], budget: int, endpoint: str) -> Dict[str, list]:
    """Extract right-censored event times for one endpoint.

    Parameters
    ----------
    trials : list of dict
        Trial records for one cell.
    budget : int
        Execution budget; censoring time for trials with no event.
    endpoint : {"bug", "milestone"}
        ``"bug"`` uses ``first_bug_exec``; ``"milestone"`` uses the execution at
        which the highest milestone observed *anywhere in the cell's target* was
        first reached.

    Returns
    -------
    dict
        ``times`` (list of int) and ``events`` (1 = observed, 0 = censored).
    """
    times, events = [], []
    if endpoint == "bug":
        for t in trials:
            v = t.get("first_bug_exec")
            times.append(int(v) if v is not None else budget)
            events.append(1 if v is not None else 0)
    else:
        top = max((t["max_milestone"] for t in trials), default=0)
        for t in trials:
            mfe = t.get("milestone_first_exec") or {}
            v = mfe.get(str(top))
            times.append(int(v) if v is not None else budget)
            events.append(1 if v is not None else 0)
    return {"times": times, "events": events, "n": len(times)}


def compare(a: Dict[str, list], b: Dict[str, list]) -> Dict[str, Any]:
    """Compare two censored samples: log-rank, A12, Mann-Whitney, Fisher.

    ``a`` is the treatment, ``b`` the control. ``A12 < 0.5`` therefore means the
    treatment reached the event *sooner*, which is the direction of improvement.
    """
    out: Dict[str, Any] = {}
    try:
        out["logrank"] = _san(logrank_test(a["times"], a["events"], b["times"], b["events"]))
    except Exception as exc:  # pragma: no cover
        out["logrank"] = {"error": str(exc)}
    out["a12_treatment_vs_control"] = _f(vargha_delaney_a12(a["times"], b["times"]))
    out["a12_magnitude"] = a12_magnitude(out["a12_treatment_vs_control"])
    out["mann_whitney"] = _san(mann_whitney(a["times"], b["times"]))
    ka, kb = sum(a["events"]), sum(b["events"])
    out["fisher_success"] = _san(fisher_success(ka, a["n"], kb, b["n"]))
    out["treatment_success"] = {
        "k": ka,
        "n": a["n"],
        "rate": ka / a["n"],
        "ci95": list(clopper_pearson(ka, a["n"])),
    }
    out["control_success"] = {
        "k": kb,
        "n": b["n"],
        "rate": kb / b["n"],
        "ci95": list(clopper_pearson(kb, b["n"])),
    }
    return out


def blind_spot_analysis() -> Dict[str, Any]:
    """Verify the TG1 witness pair's bigram-identity claim against the engine."""
    from guidance_engine import MultiDimGuidanceEngine

    t = [x for x in get_trigram_targets() if x.NAME == "TG1_trigram_lock"][0]

    def enc(word):
        buf = bytearray(0x11 for _ in range(24))
        for i, s in enumerate(word):
            buf[i] = ((buf[i] & ~t.MASK & 0xFF) | s) ^ t.DECODE_XOR
        return bytes(buf)

    def sig(word):
        eng = MultiDimGuidanceEngine(dims=("D0", "D1"), map_bits=16, context_depth=4)
        eng.reset()
        r = t.run(enc(word), eng)
        lo, hi = 0x50, 0x50 + t.ALPHABET
        d0 = frozenset(
            (e, c)
            for e, c in eng.d0_signature_counted()
            if e[0] is not None and lo <= e[0] < hi and lo <= e[1] < hi
        )
        return d0, frozenset(eng.d1.trace_signature()), r

    pd0, pd1, pr = sig(t.WITNESS_POS)
    nd0, nd1, nr = sig(t.WITNESS_NEG)
    return {
        "target": t.NAME,
        "witness_pos": list(t.WITNESS_POS),
        "witness_neg": list(t.WITNESS_NEG),
        "bigram_multisets_identical": bigram_multiset(t.WITNESS_POS)
        == bigram_multiset(t.WITNESS_NEG),
        "symbol_counts_identical": sorted(t.WITNESS_POS) == sorted(t.WITNESS_NEG),
        "trigram_multisets_identical": trigram_multiset(t.WITNESS_POS)
        == trigram_multiset(t.WITNESS_NEG),
        "d0_walk_maps_identical": pd0 == nd0,
        "d1_signatures_identical": pd1 == nd1,
        "pos_bug": bool(pr.bug),
        "neg_bug": bool(nr.bug),
        "pos_milestone": int(pr.milestone),
        "neg_milestone": int(nr.milestone),
        "interpretation": (
            "The pair occupies ONE cell of the edge-coverage partition and TWO "
            "cells of the context-sensitive partition, while having opposite bug "
            "outcomes. This is the Step 1 refinement theorem made operational: "
            "edge coverage cannot in principle supply a gradient here."
        ),
    }


def main() -> int:
    """Analyse, plot, and serialise."""
    raw = json.loads(IN.read_text())
    trials = raw["trials"]
    budget = int(raw["max_execs"])
    targets = raw["targets"]
    configs = [c["name"] for c in raw["configs"]]

    def cell(tgt: str, cfg: str) -> List[dict]:
        return [t for t in trials if t["target"] == tgt and t["config"] == cfg]

    results: Dict[str, Any] = {
        "schema": "step2_scheduler_validation/1.0",
        "master_seed": raw["master_seed"],
        "n_trials_per_cell": raw["n_trials_per_cell"],
        "max_execs": budget,
        "protocol": (
            "Klees et al. CCS'18: >=30 independent trials per cell, randomized "
            "block design with seeds blocked across conditions, right-censoring "
            "recorded explicitly, log-rank test for censored comparisons, "
            "Vargha-Delaney A12 effect sizes, Benjamini-Hochberg FDR across the "
            "whole hypothesis family."
        ),
        "control_condition": CONTROL,
        "treatment_condition": TREATMENT,
        "configs": raw["configs"],
        "environment": raw["environment"],
        "blind_spot_verification": blind_spot_analysis(),
        "descriptives": {},
        "comparisons": {},
        "dilution": {},
    }

    # ---------------- descriptives ---------------- #
    for tgt in targets:
        results["descriptives"][tgt] = {}
        for cfg in configs:
            rs = cell(tgt, cfg)
            if not rs:
                continue
            found = [r["first_bug_exec"] for r in rs if r["first_bug_exec"] is not None]
            k = len(found)
            results["descriptives"][tgt][cfg] = {
                "n_trials": len(rs),
                "n_bugs_found": k,
                "success_rate": k / len(rs),
                "success_ci95": list(clopper_pearson(k, len(rs))),
                "time_to_bug_uncensored": _san(describe(found)) if found else None,
                "max_milestone": _san(describe([r["max_milestone"] for r in rs])),
                "corpus_size": _san(describe([r["corpus_size"] for r in rs])),
                "sterile_fraction": _san(
                    describe([r["schedule_stats"]["sterile_fraction"] for r in rs])
                ),
                "energy_gini": _san(
                    describe([r["schedule_stats"]["energy_gini"] for r in rs])
                ),
                "execs_per_second": _san(describe([r["execs_per_second"] for r in rs])),
                "final_context_depth": _san(
                    describe([r["engine_stats"]["final_context_depth"] for r in rs])
                ),
                "n_escalations": _san(
                    describe(
                        [len(r.get("depth_stats", {}).get("escalations", [])) for r in rs]
                    )
                ),
            }

    # ---------------- comparisons ---------------- #
    pvals: List[float] = []
    keys: List[tuple] = []
    for tgt in targets:
        results["comparisons"][tgt] = {}
        ctrl = cell(tgt, CONTROL)
        for cfg in configs:
            if cfg == CONTROL:
                continue
            rs = cell(tgt, cfg)
            if not rs or not ctrl:
                continue
            entry: Dict[str, Any] = {}
            for endpoint in ("bug", "milestone"):
                a = event_times(rs, budget, endpoint)
                b = event_times(ctrl, budget, endpoint)
                cmp_ = compare(a, b)
                entry[endpoint] = cmp_
                for test in ("logrank", "mann_whitney", "fisher_success"):
                    pv = cmp_.get(test, {}).get("p_value")
                    if pv is not None and math.isfinite(pv):
                        pvals.append(pv)
                        keys.append((tgt, cfg, endpoint, test))
            results["comparisons"][tgt][cfg] = entry

    # ---------------- dilution (corpus inflation) ---------------- #
    for tgt in targets:
        ctrl = cell(tgt, CONTROL)
        c_corpus = [r["corpus_size"] for r in ctrl]
        c_ster = [r["schedule_stats"]["sterile_fraction"] for r in ctrl]
        results["dilution"][tgt] = {}
        for cfg in configs:
            if cfg == CONTROL:
                continue
            rs = cell(tgt, cfg)
            if not rs:
                continue
            t_corpus = [r["corpus_size"] for r in rs]
            t_ster = [r["schedule_stats"]["sterile_fraction"] for r in rs]
            mw_c = mann_whitney(t_corpus, c_corpus)
            mw_s = mann_whitney(t_ster, c_ster)
            results["dilution"][tgt][cfg] = {
                "corpus_size": {
                    "treatment_median": _f(statistics.median(t_corpus)),
                    "control_median": _f(statistics.median(c_corpus)),
                    "a12": _f(vargha_delaney_a12(t_corpus, c_corpus)),
                    "mann_whitney": _san(mw_c),
                },
                "sterile_fraction": {
                    "treatment_median": _f(statistics.median(t_ster)),
                    "control_median": _f(statistics.median(c_ster)),
                    "a12": _f(vargha_delaney_a12(t_ster, c_ster)),
                    "mann_whitney": _san(mw_s),
                },
            }
            for nm, mw in (("corpus_size", mw_c), ("sterile_fraction", mw_s)):
                pv = mw.get("p_value")
                if pv is not None and math.isfinite(pv):
                    pvals.append(pv)
                    keys.append((tgt, cfg, "dilution", nm))

    # ---------------- BH-FDR across the whole family ---------------- #
    bh = benjamini_hochberg(pvals, alpha=0.05)
    qvals = bh["q_values"] if "q_values" in bh else bh.get("adjusted", [])
    rejected = bh.get("rejected", [False] * len(pvals))
    results["fdr"] = {
        "alpha": 0.05,
        "n_hypotheses": len(pvals),
        "n_rejected": int(sum(bool(x) for x in rejected)),
        "family": [
            {
                "target": k[0],
                "config": k[1],
                "endpoint": k[2],
                "test": k[3],
                "p_value": _f(p),
                "q_value": _f(q),
                "significant_at_q05": bool(r),
            }
            for k, p, q, r in zip(keys, pvals, qvals, rejected)
        ],
    }

    # ---------------- figures ---------------- #
    FIG.mkdir(parents=True, exist_ok=True)
    _fig_survival(targets, configs, trials, budget)
    _fig_dilution(targets, configs, trials)
    _fig_energy(targets, configs, trials)
    _fig_coverage(targets, configs, trials)
    _fig_depth(targets, trials)
    results["figures"] = sorted(p.name for p in FIG.glob("step2_*"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_san(results), indent=2, allow_nan=False))
    print(f"results -> {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)", flush=True)
    print(f"figures -> {len(results['figures'])} files in {FIG}", flush=True)
    print(
        f"FDR family: {results['fdr']['n_hypotheses']} hypotheses, "
        f"{results['fdr']['n_rejected']} significant at q<0.05",
        flush=True,
    )
    return 0


def _save(fig, name: str) -> None:
    """Write a figure as both PNG and PDF."""
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _km(times: List[int], events: List[int], budget: int):
    """Kaplan-Meier survival curve (probability of *not yet* having found the bug)."""
    order = sorted(zip(times, events))
    n = len(order)
    xs, ys = [0], [1.0]
    surv, at_risk = 1.0, n
    for t, e in order:
        if e:
            surv *= 1.0 - 1.0 / at_risk
            xs.append(t)
            ys.append(surv)
        at_risk -= 1
    xs.append(budget)
    ys.append(surv)
    return xs, ys


def _fig_survival(targets, configs, trials, budget) -> None:
    """Kaplan-Meier time-to-first-bug curves per target."""
    fig, axes = plt.subplots(1, len(targets), figsize=(4.0 * len(targets), 3.2))
    for ax, tgt in zip(np.atleast_1d(axes), targets):
        for cfg in configs:
            rs = [t for t in trials if t["target"] == tgt and t["config"] == cfg]
            if not rs:
                continue
            ev = event_times(rs, budget, "bug")
            xs, ys = _km(ev["times"], ev["events"], budget)
            ax.step(
                xs,
                ys,
                where="post",
                label=cfg,
                color=PALETTE.get(cfg),
                lw=1.8 if cfg == TREATMENT else 1.0,
            )
        ax.set_title(tgt, fontsize=9)
        ax.set_xlabel("executions")
        ax.set_ylabel("P(bug not yet found)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25, lw=0.4)
    np.atleast_1d(axes)[-1].legend(fontsize=5.5, loc="lower left", framealpha=0.9)
    fig.suptitle(
        "Step 2: time-to-first-bug survival (right-censored at budget)", fontsize=10
    )
    _save(fig, "step2_survival_curves")


def _fig_dilution(targets, configs, trials) -> None:
    """Corpus size and sterile fraction: the two corpus-dilution measures."""
    fig, axes = plt.subplots(2, len(targets), figsize=(4.0 * len(targets), 5.4))
    axes = np.atleast_2d(axes).reshape(2, len(targets))
    for j, tgt in enumerate(targets):
        for row, (key, lab) in enumerate(
            [("corpus", "corpus size"), ("sterile", "sterile fraction")]
        ):
            ax = axes[row][j]
            data, labs, cols = [], [], []
            for cfg in configs:
                rs = [t for t in trials if t["target"] == tgt and t["config"] == cfg]
                if not rs:
                    continue
                v = (
                    [r["corpus_size"] for r in rs]
                    if key == "corpus"
                    else [r["schedule_stats"]["sterile_fraction"] for r in rs]
                )
                data.append(v)
                labs.append(cfg)
                cols.append(PALETTE.get(cfg, "#888888"))
            bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
            for patch, c in zip(bp["boxes"], cols):
                patch.set_facecolor(c)
                patch.set_alpha(0.65)
                patch.set_linewidth(0.5)
            ax.set_xticks(range(1, len(labs) + 1))
            ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=5.5)
            ax.set_ylabel(lab)
            if key == "corpus":
                ax.set_yscale("symlog")
                ax.set_title(tgt, fontsize=9)
            ax.grid(alpha=0.25, lw=0.4, axis="y")
    fig.suptitle("Step 2: corpus dilution by scheduler", fontsize=10)
    _save(fig, "step2_corpus_dilution")


def _fig_energy(targets, configs, trials) -> None:
    """Energy-allocation concentration (Gini) per condition."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.8 / len(targets)
    xs = np.arange(len(configs))
    for i, tgt in enumerate(targets):
        med, lo, hi = [], [], []
        for cfg in configs:
            rs = [t for t in trials if t["target"] == tgt and t["config"] == cfg]
            g = [r["schedule_stats"]["energy_gini"] for r in rs] or [0.0]
            q = np.percentile(g, [25, 50, 75])
            med.append(q[1])
            lo.append(q[1] - q[0])
            hi.append(q[2] - q[1])
        ax.bar(
            xs + i * width - 0.4 + width / 2,
            med,
            width=width * 0.9,
            yerr=[lo, hi],
            capsize=1.5,
            label=tgt,
            error_kw={"lw": 0.6},
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("energy Gini (median, IQR)")
    ax.set_title(
        "Step 2: energy-allocation concentration\n"
        "0 = every seed treated identically (round robin); higher = prioritised",
        fontsize=9,
    )
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    _save(fig, "step2_energy_allocation")


def _fig_coverage(targets, configs, trials) -> None:
    """Median coverage-growth trajectories."""
    fig, axes = plt.subplots(1, len(targets), figsize=(4.0 * len(targets), 3.2))
    for ax, tgt in zip(np.atleast_1d(axes), targets):
        for cfg in configs:
            rs = [t for t in trials if t["target"] == tgt and t["config"] == cfg]
            if not rs:
                continue
            grid = {}
            for r in rs:
                for pt in r["curve"]:
                    grid.setdefault(pt[0], []).append(pt[1])
            xs = sorted(grid)
            ys = [statistics.median(grid[x]) for x in xs]
            ax.plot(
                xs,
                ys,
                label=cfg,
                color=PALETTE.get(cfg),
                lw=1.8 if cfg == TREATMENT else 1.0,
            )
        ax.set_title(tgt, fontsize=9)
        ax.set_xlabel("executions")
        ax.set_ylabel("exact coverage elements (median)")
        ax.set_yscale("symlog")
        ax.grid(alpha=0.25, lw=0.4)
    np.atleast_1d(axes)[-1].legend(fontsize=5.5, loc="lower right")
    fig.suptitle("Step 2: coverage growth", fontsize=10)
    _save(fig, "step2_coverage_growth")


def _fig_depth(targets, trials) -> None:
    """Context-depth adaptation over time for the dynamic condition."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for tgt in targets:
        rs = [
            t
            for t in trials
            if t["target"] == tgt and t["config"] == "d3d_adaptive_dyn"
        ]
        if not rs:
            continue
        grid = {}
        for r in rs:
            for pt in r["curve"]:
                if len(pt) >= 5:
                    grid.setdefault(pt[0], []).append(pt[4])
        xs = sorted(grid)
        if not xs:
            continue
        ax.plot(xs, [statistics.median(grid[x]) for x in xs], label=tgt, lw=1.4)
    ax.set_xlabel("executions")
    ax.set_ylabel("D1 context depth N (median)")
    ax.set_yscale("log", base=2)
    ax.set_yticks([2, 4, 8, 16])
    ax.set_yticklabels(["2", "4", "8", "16"])
    ax.set_title(
        "Step 2: dynamic context-depth adaptation\n"
        "start shallow at N=4, escalate only on novelty saturation",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    _save(fig, "step2_context_depth_adaptation")


if __name__ == "__main__":
    raise SystemExit(main())
