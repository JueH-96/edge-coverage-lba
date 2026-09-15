"""
Publication-quality figures for the Step 1 validation.

Reads ``results/step1_guidance_validation.json`` and
``data/bug_finding_trials.json`` and writes PNG + PDF pairs to ``figures/``.

Figure integrity rules followed here: bars start at zero, uncertainty is always
named (95% Clopper-Pearson for proportions, bootstrap or IQR elsewhere), censored
observations are drawn as censored rather than silently substituted, colour is never
the only channel carrying information, and no axis is truncated to exaggerate an
effect.

    uv run python workflow/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402


def _sci_tick(v: float) -> str:
    """Compact tick label for log axes: 40k -> '4e4', 100000 -> '1e5'."""
    if v <= 0:
        return ""
    exp = int(np.floor(np.log10(v)))
    mant = v / (10.0 ** exp)
    if abs(mant - round(mant)) < 1e-9:
        mant_s = f"{int(round(mant))}"
    else:
        mant_s = f"{mant:.1f}"
    if mant_s == "1":
        return rf"$10^{{{exp}}}$"
    return rf"${mant_s}\!\times\!10^{{{exp}}}$"

# --- artifact-evaluation path bootstrap (replaces the session-root anchor) --- #
import sys as _ae_sys  # noqa: E402
from pathlib import Path as _AEPath  # noqa: E402

AE_ROOT = _AEPath(__file__).resolve().parent.parent
if str(AE_ROOT) not in _ae_sys.path:
    _ae_sys.path.insert(0, str(AE_ROOT))
import ae_paths as _ae_paths  # noqa: E402,F401  (registers src/ and targets/)

SESSION_DIR = AE_ROOT
# --- end bootstrap --------------------------------------------------------- #
FIG_DIR = SESSION_DIR / "figures"
RESULTS = SESSION_DIR / "results" / "step1_guidance_validation.json"
TRIALS = SESSION_DIR / "data" / "step1_trials.json"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "figure.dpi": 120,
    }
)

#: Colour-blind-safe palette (Okabe-Ito); every series also gets a distinct marker
#: and line style so colour is never the sole channel.
PALETTE = {
    "blind_random": "#999999",
    "edge_only": "#000000",
    "d0_d1_context": "#E69F00",
    "d0_d2_valuerange": "#0072B2",
    "d0_d3_statemachine": "#009E73",
    "full_3d": "#CC79A7",
}
MARKERS = {
    "blind_random": "x",
    "edge_only": "o",
    "d0_d1_context": "s",
    "d0_d2_valuerange": "^",
    "d0_d3_statemachine": "D",
    "full_3d": "v",
}
STYLES = {
    "blind_random": (0, (1, 1)),
    "edge_only": "-",
    "d0_d1_context": (0, (4, 1.5)),
    "d0_d2_valuerange": (0, (3, 1, 1, 1)),
    "d0_d3_statemachine": (0, (5, 1, 1, 1, 1, 1)),
    "full_3d": (0, (2, 1)),
}
SHORT = {
    "blind_random": "blind random",
    "edge_only": "AFL edges (D0)",
    "d0_d1_context": "D0+D1 context",
    "d0_d2_valuerange": "D0+D2 value",
    "d0_d3_statemachine": "D0+D3 state",
    "full_3d": "full 3D",
}


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{stem}.png|.pdf", flush=True)


def fig_success_rates(rep: Dict[str, Any]) -> None:
    """Bug-discovery success rate per configuration, with exact binomial CIs."""
    targets = list(rep["bug_finding"]["per_target"])
    configs = list(rep["configurations"])
    fig, axes = plt.subplots(1, len(targets), figsize=(7.2, 2.6), sharey=True)
    for ax, tname in zip(np.atleast_1d(axes), targets):
        tv = rep["bug_finding"]["per_target"][tname]
        xs = np.arange(len(configs))
        rates, los, his = [], [], []
        for c in configs:
            v = tv["configs"][c]
            rates.append(v["success_rate"])
            lo, hi = v["success_rate_ci95"]
            los.append(max(0.0, v["success_rate"] - lo))
            his.append(max(0.0, hi - v["success_rate"]))
        ax.bar(
            xs, rates, color=[PALETTE[c] for c in configs], width=0.7,
            edgecolor="black", linewidth=0.4,
        )
        ax.errorbar(
            xs, rates, yerr=[los, his], fmt="none", ecolor="black",
            elinewidth=0.7, capsize=2.5,
        )
        for x, c in zip(xs, configs):
            v = tv["configs"][c]
            ax.text(x, 0.02, f"{v['n_bugs_found']}/{v['n_trials']}", ha="center",
                    va="bottom", fontsize=5.5, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[c] for c in configs], rotation=40, ha="right", fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{tname}\n(budget {tv['budget_execs']:,} execs)", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("bug-discovery rate\n(95% Clopper-Pearson CI)")
    fig.suptitle(
        "Seeded-bug discovery rate over 30 blocked trials per cell", fontsize=9, y=1.06
    )
    _save(fig, "fig1_bug_discovery_rate")


def fig_time_to_bug_cdf(rep: Dict[str, Any], trials: List[Dict[str, Any]]) -> None:
    """Empirical CDF of executions-to-bug, with censored trials shown explicitly."""
    targets = list(rep["bug_finding"]["per_target"])
    configs = list(rep["configurations"])
    fig, axes = plt.subplots(1, len(targets), figsize=(8.0, 2.6))
    fig.subplots_adjust(wspace=0.30)
    for ax, tname in zip(np.atleast_1d(axes), targets):
        budget = rep["bug_finding"]["per_target"][tname]["budget_execs"]
        first_event = budget
        n_censored_any = False
        for c in configs:
            rs = [r for r in trials if r["target"] == tname and r["config"] == c]
            if not rs:
                continue
            n = len(rs)
            found = sorted(r["first_bug_exec"] for r in rs if r["first_bug_exec"])
            if found:
                first_event = min(first_event, found[0])
            xs = [0] + found + [budget]
            ys = [0] + [(i + 1) / n for i in range(len(found))] + [len(found) / n]
            ax.step(
                xs, ys, where="post", color=PALETTE[c], linestyle=STYLES[c],
                linewidth=1.3, label=SHORT[c],
            )
            if found:
                step = max(1, len(found) // 6)
                pts = found[::step]
                ax.plot(pts, [(found.index(v) + 1) / n for v in pts],
                        MARKERS[c], color=PALETTE[c], markersize=3, linestyle="none")
            if len(found) < n:  # censored: plateau end marked with a cross-tick
                n_censored_any = True
                ax.plot([budget], [len(found) / n], marker="+", color=PALETTE[c],
                        markersize=6, markeredgewidth=1.3, clip_on=False)
        # start the axis just below the earliest event instead of at 1 execution
        ax.set_xscale("log")
        ax.set_xlim(max(1.0, first_event * 0.4), budget * 1.25)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("executions (log scale)")
        # When the visible decade span is narrow, matplotlib's default minor
        # log labels collide. Thin them to at most four well-separated ticks.
        lo, hi = ax.get_xlim()
        if hi / lo < 20.0:
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _p: _sci_tick(v))
            )
        title = tname if not n_censored_any else f"{tname} (+ = censored at budget)"
        ax.set_title(title, fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("fraction of trials\nwith bug found")
    np.atleast_1d(axes)[-1].legend(fontsize=5.5, loc="upper left")
    fig.suptitle(
        "Time-to-bug distribution; vertical tick = trials still censored at the budget",
        fontsize=9, y=1.06,
    )
    _save(fig, "fig2_time_to_bug_cdf")


def fig_progress_curves(rep: Dict[str, Any], trials: List[Dict[str, Any]]) -> None:
    """Median milestone progress vs executions, with IQR bands."""
    targets = list(rep["bug_finding"]["per_target"])
    configs = list(rep["configurations"])
    fig, axes = plt.subplots(1, len(targets), figsize=(7.6, 2.6))
    for ax, tname in zip(np.atleast_1d(axes), targets):
        tv = rep["bug_finding"]["per_target"][tname]
        for c in configs:
            rs = [r for r in trials if r["target"] == tname and r["config"] == c]
            if not rs:
                continue
            # resample every trial's step curve onto a common execution grid
            budget = tv["budget_execs"]
            grid = np.unique(np.geomspace(100, budget, 40).astype(int))
            mat = np.zeros((len(rs), grid.size))
            for i, r in enumerate(rs):
                if not r["curve"]:
                    continue
                cx = np.array([p[0] for p in r["curve"]])
                cy = np.array([p[3] for p in r["curve"]])
                idx = np.searchsorted(cx, grid, side="right") - 1
                mat[i] = np.where(idx >= 0, cy[np.clip(idx, 0, len(cy) - 1)], 0)
            med = np.median(mat, axis=0)
            q1 = np.percentile(mat, 25, axis=0)
            q3 = np.percentile(mat, 75, axis=0)
            ax.plot(grid, med, color=PALETTE[c], linestyle=STYLES[c], linewidth=1.3,
                    label=SHORT[c])
            ax.fill_between(grid, q1, q3, color=PALETTE[c], alpha=0.12, linewidth=0)
        ax.axhline(tv["max_milestone_possible"], color="red", linestyle=":", linewidth=0.8)
        ax.text(
            ax.get_xlim()[0], tv["max_milestone_possible"], " bug", color="red",
            fontsize=5.5, va="bottom",
        )
        ax.set_xscale("log")
        ax.set_xlabel("executions (log scale)")
        ax.set_title(f"{tname} (targets {tv['targeted_dimension']})", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("milestone reached\n(median, IQR band)")
    np.atleast_1d(axes)[-1].legend(fontsize=5.5, loc="lower right")
    fig.suptitle("Progress toward the seeded bug (30 trials per configuration)",
                 fontsize=9, y=1.06)
    _save(fig, "fig3_progress_curves")


def fig_partition_refinement(rep: Dict[str, Any]) -> None:
    """Partition classes and entropy per dimension (the benefit side of the ledger)."""
    pd_ = rep["partition_discrimination"]["per_target"]
    targets = list(pd_)
    keys = ["D0", "D0_with_hit_counts", "D1", "D2", "D3", "composite_3d"]
    labels = ["D0\nedges", "D0\n+counts", "D1\ncontext", "D2\nvalue", "D3\nstate", "3D\ncomposite"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    width = 0.8 / len(targets)
    xs = np.arange(len(keys))
    hatches = ["", "//", ".."]
    for i, t in enumerate(targets):
        cls = [pd_[t]["partitions"][k]["n_classes"] for k in keys]
        ent = [pd_[t]["partitions"][k]["partition_entropy_bits"] for k in keys]
        off = (i - (len(targets) - 1) / 2) * width
        axes[0].bar(xs + off, cls, width=width, label=t, edgecolor="black",
                    linewidth=0.4, hatch=hatches[i % 3])
        axes[1].bar(xs + off, ent, width=width, label=t, edgecolor="black",
                    linewidth=0.4, hatch=hatches[i % 3])
    for ax, ylab, logy in ((axes[0], "distinct coverage classes", True),
                           (axes[1], "partition entropy H(Pi) [bits]", False)):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel(ylab, fontsize=7)
        if logy:
            ax.set_yscale("log")
    # Legend outside the axes: inside, it overlapped the tallest T1/T3 bars.
    axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   ncol=3, frameon=False)
    fig.suptitle(
        "Partition refinement on a shared corpus (~1500 inputs/target): "
        "finer partition = more behaviours distinguished",
        fontsize=8.5, y=1.05,
    )
    _save(fig, "fig4_partition_refinement")


def fig_overhead(rep: Dict[str, Any]) -> None:
    """Per-execution overhead per dimension, relative to the D0 baseline."""
    ov = rep["overhead"]["per_target"]
    targets = list(ov)
    keys = [
        "uninstrumented", "D0|deployment", "D1|deployment", "D2|deployment",
        "D3|deployment", "D0+D1|deployment", "D0+D2|deployment", "D0+D3|deployment",
        "D0+D1+D2+D3|deployment", "D0+D1+D2+D3|analysis",
    ]
    labels = ["none", "D0", "D1", "D2", "D3", "D0+D1", "D0+D2", "D0+D3",
              "3D\ndeploy", "3D\nanalysis"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    xs = np.arange(len(keys))
    width = 0.8 / len(targets)
    hatches = ["", "//", ".."]
    for i, t in enumerate(targets):
        vals, errs = [], [[], []]
        for k in keys:
            v = ov[t].get(k)
            if v is None:
                vals.append(np.nan)
                errs[0].append(0)
                errs[1].append(0)
                continue
            vals.append(v["us_per_exec_median"])
            lo, hi = v["us_per_exec_iqr"]
            errs[0].append(max(0.0, v["us_per_exec_median"] - lo))
            errs[1].append(max(0.0, hi - v["us_per_exec_median"]))
        off = (i - (len(targets) - 1) / 2) * width
        ax.bar(xs + off, vals, width=width, label=t, edgecolor="black",
               linewidth=0.4, hatch=hatches[i % 3])
        ax.errorbar(xs + off, vals, yerr=errs, fmt="none", ecolor="black",
                    elinewidth=0.6, capsize=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("us per execution\n(median, IQR)", fontsize=7)
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    fig.suptitle(
        "Tracking overhead (pure-Python reference implementation; "
        "ratios transfer, absolute values do not)",
        fontsize=8.5, y=1.03,
    )
    _save(fig, "fig5_overhead")


def fig_context_depth_sweep(rep: Dict[str, Any]) -> None:
    """Context window depth N against effectiveness and state-space cost."""
    sw = rep["context_depth_sweep"]["per_depth"]
    depths = sorted(int(k) for k in sw)
    ctx = [sw[str(d)]["distinct_contexts_exact"]["median"] for d in depths]
    corp = [sw[str(d)]["corpus_size"]["median"] for d in depths]
    eps = [sw[str(d)]["mean_exec_per_second"] for d in depths]

    # Every depth reaches 100% success, so time-to-bug is the discriminating
    # outcome; plot it against the edge-coverage baseline instead of the flat rate.
    med, lo_e, hi_e, sig = [], [], [], []
    for d in depths:
        e = sw[str(d)]["execs_to_bug_among_successes"]
        m = e.get("median", np.nan)
        cl, ch = e.get("median_ci95", [np.nan, np.nan])
        med.append(m)
        lo_e.append(max(0.0, m - cl))
        hi_e.append(max(0.0, ch - m))
        q = sw[str(d)].get("vs_edge_only_baseline", {}).get("logrank_time_to_bug_q_value")
        sig.append(q is not None and q == q and q <= 0.05)
    base_med = None
    bf = rep["bug_finding"]["per_target"].get(rep["context_depth_sweep"]["target"])
    if bf:
        base_med = bf["configs"]["edge_only"]["execs_to_bug_among_successes"].get("median")

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.5))
    fig.subplots_adjust(wspace=0.42)
    axes[0].errorbar(
        depths, med, yerr=[lo_e, hi_e], marker="o", color="#0072B2",
        linewidth=1.2, capsize=2.5, markersize=4, zorder=3,
    )
    if base_med:
        axes[0].axhline(base_med, color="black", linestyle="-", linewidth=1.0,
                        label=f"edge_only baseline ({base_med:,.0f})")
        axes[0].legend(fontsize=5.5, loc="upper left")
    for d, m, s in zip(depths, med, sig):
        if s:  # significantly different from the baseline after FDR correction
            axes[0].annotate("*", (d, m), fontsize=11, color="#0072B2",
                             ha="center", va="top", xytext=(0, -9),
                             textcoords="offset points")
    axes[0].set_ylim(0, max(max(med), base_med or 0) * 1.25)
    axes[0].set_ylabel("median executions to bug\n(95% bootstrap CI)", fontsize=7)

    axes[1].plot(depths, ctx, marker="s", color="#E69F00", linewidth=1.2,
                 label="distinct contexts")
    axes[1].plot(depths, corp, marker="D", color="#009E73", linewidth=1.2,
                 linestyle="--", label="corpus entries")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("count (median)", fontsize=7)
    axes[1].legend(fontsize=6)

    axes[2].plot(depths, eps, marker="^", color="#CC79A7", linewidth=1.2)
    axes[2].set_ylim(0, max(eps) * 1.15)  # zero-based: do not exaggerate the drop
    axes[2].set_ylabel("executions / second", fontsize=7)
    drop = 100 * (1 - eps[-1] / eps[0]) if eps[0] else 0.0
    axes[2].annotate(f"-{drop:.0f}% from N=1 to N={depths[-1]}",
                     (depths[-1], eps[-1]), fontsize=6, ha="right",
                     xytext=(-4, 12), textcoords="offset points")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_xlabel("context window depth N", fontsize=7)
    fig.suptitle(
        f"Calling-context depth sweep on {rep['context_depth_sweep']['target']} "
        "(30 trials/depth, all reach 100% success):\n"
        "* = log-rank q <= 0.05 vs the edge-coverage baseline on the same seeds",
        fontsize=8, y=1.13,
    )
    _save(fig, "fig6_context_depth_sweep")


def fig_collisions(rep: Dict[str, Any]) -> None:
    """Measured vs birthday-bound hash collision rate across map sizes."""
    dc = rep["dimensionality_and_collisions"]["per_target"]
    targets = list(dc)
    fig, axes = plt.subplots(1, len(targets), figsize=(7.6, 2.5), sharey=True)
    dims = ["D0", "D1", "D2", "D3"]
    colors = {"D0": "#000000", "D1": "#E69F00", "D2": "#0072B2", "D3": "#009E73"}
    marks = {"D0": "o", "D1": "s", "D2": "^", "D3": "D"}
    for ax, t in zip(np.atleast_1d(axes), targets):
        bybits = dc[t]["by_map_bits"]
        bits = sorted(int(b) for b in bybits)
        for d in dims:
            meas = [bybits[str(b)][d]["collision_rate"] for b in bits]
            ax.plot(bits, meas, marker=marks[d], color=colors[d], linewidth=1.2,
                    markersize=4, label=f"{d} measured")
            exp = []
            for b in bits:
                k = bybits[str(b)][d]["exact_elements"]
                exp.append(min(1.0, (k / (2 ** (b + 1)))) if k else 0.0)
            ax.plot(bits, exp, color=colors[d], linewidth=0.8, linestyle=":", alpha=0.7)
        ax.set_xticks(bits)
        ax.set_xlabel("map size (log2 slots)", fontsize=7)
        ax.set_title(t, fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("collision rate\n(exact - mapped)/exact", fontsize=7)
    np.atleast_1d(axes)[0].legend(fontsize=5.5)
    fig.suptitle(
        "Bitmap collision rate per dimension; dotted = birthday expectation "
        "k$^2$/2$^{b+1}$ per element, clipped at 1",
        fontsize=8.5, y=1.05,
    )
    _save(fig, "fig7_collision_rates")


def fig_state_graph(rep: Dict[str, Any]) -> None:
    """Inferred implicit state-machine graph for T3 (spring layout, no networkx)."""
    import guidance_engine as ge_mod  # local import keeps module import cheap
    from fuzz_harness import SEED_INPUT as SEED, Mutator
    from test_targets import T3StateMachineTarget
    import random

    tgt = T3StateMachineTarget()
    eng = ge_mod.MultiDimGuidanceEngine(dims=("D0", "D3"), map_bits=16)
    rng = random.Random(4242)
    mut = Mutator(rng)
    corpus = [SEED]
    data = SEED
    for i in range(20000):
        data = mut.mutate(corpus[i % len(corpus)], splice_pool=corpus)
        eng.reset()
        tgt.run(data, eng)
        if eng.evaluate(commit=True).is_interesting:
            corpus.append(data)

    g = eng.d3.graph_export()
    nodes = g["nodes"]
    LIMIT = 48
    if len(nodes) > LIMIT:
        # Keep the most-visited subgraph legible, but ALWAYS retain the
        # use-after-free state and its direct predecessors - it is rare, so a
        # pure degree ranking prunes exactly the node the figure is about.
        deg: Dict[str, int] = {}
        for e in g["edges"]:
            deg[e["src"]] = deg.get(e["src"], 0) + e["count"]
            deg[e["dst"]] = deg.get(e["dst"], 0) + e["count"]
        must = {n for n in nodes if "USE_AFTER_FREE" in n}
        must |= {e["src"] for e in g["edges"] if e["dst"] in must}
        rest = sorted((n for n in nodes if n not in must), key=lambda n: -deg.get(n, 0))
        nodes = list(must) + rest[: max(0, LIMIT - len(must))]
    keep = set(nodes)
    edges = [e for e in g["edges"] if e["src"] in keep and e["dst"] in keep]

    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rs = np.random.default_rng(7)
    pos = rs.normal(scale=1.0, size=(n, 2))
    adj = np.zeros((n, n))
    for e in edges:
        adj[idx[e["src"]], idx[e["dst"]]] = 1
        adj[idx[e["dst"]], idx[e["src"]]] = 1
    # Fruchterman-Reingold style layout. Distances are floored and the per-node
    # displacement is clipped, otherwise coincident nodes make the 1/d^2 repulsion
    # term overflow and the layout collapses to NaN.
    MIN_DIST, MAX_STEP = 0.05, 0.5
    for step in range(400):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.sqrt((delta ** 2).sum(-1)), MIN_DIST)
        unit = delta / dist[..., None]
        rep_f = unit / dist[..., None] * 0.55
        att_f = -unit * dist[..., None] * adj[..., None] * 0.05
        np.fill_diagonal(rep_f[:, :, 0], 0.0)
        np.fill_diagonal(rep_f[:, :, 1], 0.0)
        disp = (rep_f + att_f).sum(axis=1)
        norm = np.linalg.norm(disp, axis=1, keepdims=True)
        disp = disp * np.minimum(1.0, MAX_STEP / np.maximum(norm, 1e-12))
        pos += disp * (0.9 * (1 - step / 400) + 0.05)
        pos -= pos.mean(axis=0)
        span = np.abs(pos).max()
        if span > 1e3:  # keep the layout in a sane numeric range
            pos /= span / 10.0
    assert np.isfinite(pos).all(), "layout diverged"

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    maxc = max((e["count"] for e in edges), default=1)
    for e in edges:
        a, b = idx[e["src"]], idx[e["dst"]]
        ax.annotate(
            "", xy=pos[b], xytext=pos[a],
            arrowprops=dict(
                arrowstyle="-|>", color="#0072B2",
                alpha=0.15 + 0.55 * (e["count"] / maxc),
                linewidth=0.3 + 1.4 * (e["count"] / maxc),
                shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=0.12",
            ),
        )
    is_bug = [("USE_AFTER_FREE" in nd) for nd in nodes]
    ax.scatter(
        pos[:, 0], pos[:, 1],
        s=[70 if b else 26 for b in is_bug],
        c=["#D55E00" if b else "#F0E442" for b in is_bug],
        edgecolors="black", linewidths=0.5, zorder=3,
        marker="o",
    )
    for i, b in enumerate(is_bug):
        if b:
            ax.annotate("use-after-free", pos[i], fontsize=6, color="#D55E00",
                        xytext=(6, 6), textcoords="offset points", weight="bold")
    st = eng.d3.stats()
    ax.set_axis_off()
    ax.set_title(
        f"T3 inferred implicit state machine\n"
        f"{st['n_abstract_states']} abstract states, {st['n_transitions']} transitions, "
        f"largest SCC = {st['largest_scc_size']}, "
        f"transition entropy = {st['transition_entropy_bits']:.2f} bits",
        fontsize=8,
    )
    _save(fig, "fig8_state_machine_graph")


def main() -> int:
    """Generate every figure."""
    if not RESULTS.exists():
        print(f"missing {RESULTS}; run test_step1_guidance.py --full first",
              file=sys.stderr)
        return 1
    rep = json.loads(RESULTS.read_text())
    trials = json.loads(TRIALS.read_text()) if TRIALS.exists() else []
    print("generating figures ...", flush=True)
    fig_success_rates(rep)
    if trials:
        fig_time_to_bug_cdf(rep, trials)
        fig_progress_curves(rep, trials)
    fig_partition_refinement(rep)
    fig_overhead(rep)
    fig_context_depth_sweep(rep)
    fig_collisions(rep)
    fig_state_graph(rep)
    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
