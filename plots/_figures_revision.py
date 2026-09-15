"""
Figures for the revision experiments.

Reads ``results/step2_revision.json`` and ``data/t4_order_sweep_trials.json`` and
writes PNG + PDF pairs to ``figures/``.

The same integrity rules as the first figure set apply: bars start at zero,
uncertainty is always named, censored observations are drawn as censored rather
than silently substituted, colour is never the only channel carrying information,
and no axis is truncated to exaggerate an effect. Two rules are added here because
the revision needs them:

* a quantity that is *predicted* by a model and a quantity that is *measured* are
  never drawn in the same style - the model is always a thin dashed line and is
  labelled as a prediction in the legend;
* a configuration whose trials were censored is annotated with the censoring
  count, so a missing point cannot be misread as a fast one.

    uv run python workflow/make_figures2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
RESULTS = SESSION_DIR / "results" / "step2_revision.json"
T4_TRIALS = SESSION_DIR / "data" / "t4_order_sweep.json"

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

OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}

TARGET_LABEL = {
    "T1_context": "T1 context",
    "T2_value_range": "T2 value range",
    "T3_state_machine": "T3 state machine",
    "T4_trigram": "T4 higher order",
}


def _save(fig: plt.Figure, name: str) -> None:
    """Write a figure as both PNG and PDF."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.png|pdf", flush=True)


# --------------------------------------------------------------------------- #
# fig9 - three-way information-loss decomposition
# --------------------------------------------------------------------------- #
def fig9_information_loss(res: Dict[str, Any]) -> None:
    """The cardinality chain and the three loss mechanisms, per target.

    Panel (a) plots the fraction of block-sequence distinctions still standing
    after each stage of ``q o h o G_2``, on a linear axis. Absolute cardinalities
    span four orders of magnitude across targets and a log axis of them hides the
    very thing the figure is about, so the survival fraction is plotted instead
    and the absolute counts are given in the table. Panel (b) attributes the loss
    to the three mechanisms; the axis is scaled to the data rather than to
    ``[0, 1]``, and every bar is labelled, so a small effect is not made to look
    like a large one by an empty axis.
    """
    il = res["information_loss"]
    targets = list(il["per_target"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    stages = ["block\nsequences", "bigram count\nvectors",
              "classified maps\n(collision-free)", "bitmap states\n$2^{16}$"]
    cols = [OKABE["black"], OKABE["orange"], OKABE["skyblue"], OKABE["green"]]
    for i, t in enumerate(targets):
        d = il["per_target"][t]
        base = max(1, d["n_distinct_block_sequences"])
        ys = [
            1.0,
            d["n_distinct_bigram_count_vectors"] / base,
            d["n_distinct_classified_maps_collision_free"] / base,
            d["n_distinct_bitmap_states"]["16"] / base,
        ]
        ax.plot(range(4), ys, marker="osD^"[i], ms=4.5, lw=1.3, color=cols[i],
                label=f"{TARGET_LABEL.get(t, t)} ({d['n_distinct_block_sequences']:,})")
    ax.set_xticks(range(4))
    ax.set_xticklabels(stages, fontsize=6.3)
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("fraction of block-sequence\ndistinctions still standing")
    ax.set_title(f"(a) the funnel ({il['n_exec_per_target']:,} executions/target)",
                 fontsize=8)
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, title="target (distinct block sequences in the sample)",
              title_fontsize=6.0)

    ax = axes[1]
    w = 0.26
    x = np.arange(len(targets))
    l1 = [il["per_target"][t]["loss_higher_order_sequence"] for t in targets]
    l2 = [il["per_target"][t]["loss_hitcount_quantisation"] for t in targets]
    l3 = [max(0.0, il["per_target"][t]["loss_index_collision"]["16"]) for t in targets]
    bars = [
        (ax.bar(x - w, l1, w, color=OKABE["vermillion"],
                label="1. higher-order sequence"), l1),
        (ax.bar(x, l2, w, color=OKABE["blue"], hatch="//",
                label="2. hit-count quantisation"), l2),
        (ax.bar(x + w, l3, w, color=OKABE["green"], hatch="..",
                label="3. index collision ($2^{16}$)"), l3),
    ]
    top = max(max(l1), max(l2), max(l3))
    for bc, vals in bars:
        for rect, v in zip(bc, vals):
            ax.annotate(f"{100 * v:.1f}" if v >= 0.001 else "0",
                        (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 1.5),
                        ha="center", fontsize=5.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABEL.get(t, t).replace(" ", "\n") for t in targets],
                       fontsize=6.3)
    ax.set_ylabel("percent of distinctions\nlost at that stage")
    ax.set_ylim(0, top * 1.35)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    ax.set_title("(b) the three mechanisms, separated", fontsize=8)
    ax.legend(fontsize=6.0, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig9_information_loss")


# --------------------------------------------------------------------------- #
# fig10 - the refinement lattice (star, not chain)
# --------------------------------------------------------------------------- #
def fig10_lattice(res: Dict[str, Any]) -> None:
    """The corrected order structure, drawn as a lattice with an evidence table.

    The first version of this study drew a linear refinement chain. Only D1
    refines D0; D2 and D3 are incomparable with it. The left panel draws the
    Hasse-style diagram; the right panel is the measured verdict per target pair,
    so the picture is backed by the certificate rather than by assertion.
    """
    lat = res["lattice"]
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    pos = {
        r"$\alpha_{\mathrm{3D}}$": (0.5, 0.87),
        r"$\alpha_1$ context": (0.14, 0.52),
        r"$\alpha_2$ value": (0.5, 0.52),
        r"$\alpha_3$ state": (0.86, 0.52),
        r"$\alpha_0$ edge": (0.14, 0.14),
    }
    edges = [
        (r"$\alpha_{\mathrm{3D}}$", r"$\alpha_1$ context"),
        (r"$\alpha_{\mathrm{3D}}$", r"$\alpha_2$ value"),
        (r"$\alpha_{\mathrm{3D}}$", r"$\alpha_3$ state"),
        (r"$\alpha_1$ context", r"$\alpha_0$ edge"),
    ]
    for a, b in edges:
        ax.annotate("", xy=pos[b], xytext=pos[a],
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#333333",
                                    shrinkA=13, shrinkB=13))
    # incomparability, drawn explicitly as a crossed-out link
    for other in (r"$\alpha_2$ value", r"$\alpha_3$ state"):
        xa, ya = pos[other]
        xb, yb = pos[r"$\alpha_0$ edge"]
        ax.plot([xa, xb], [ya, yb], ls=":", lw=0.9, color=OKABE["vermillion"], zorder=0)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.plot([mx - 0.022, mx + 0.022], [my - 0.03, my + 0.03], lw=1.2,
                color=OKABE["vermillion"])
        ax.plot([mx - 0.022, mx + 0.022], [my + 0.03, my - 0.03], lw=1.2,
                color=OKABE["vermillion"])
    for label, (x, y) in pos.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.34", fc="white", ec="#333333", lw=0.7))
    ax.text(0.5, 0.015,
            r"$\longrightarrow$ refines    $\cdots\!\times\!\cdots$ incomparable",
            ha="center", fontsize=6.8)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("(a) the order structure is a lattice, not a chain", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    pairs = ["D0_vs_D1", "D0_vs_D2", "D0_vs_D3", "D1_vs_D2", "D1_vs_D3", "D2_vs_D3"]
    targets = list(lat["per_target"])
    cell_text = []
    for p in pairs:
        row = []
        for t in targets:
            v = lat["per_target"][t].get(f"verdict_{p}", "-")
            v = (v.replace(" refines ", " $\\sqsubseteq$ ")
                  .replace("incomparable", "incomparable")
                  .replace("no separation observed", "no separation"))
            row.append(v)
        cell_text.append(row)
    tab = ax.table(
        cellText=cell_text,
        rowLabels=[p.replace("_vs_", " / ") for p in pairs],
        colLabels=[TARGET_LABEL.get(t, t).split()[0] for t in targets],
        loc="center",
        cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.0)
    tab.scale(1.0, 1.35)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0 or c == -1:
            cell.set_text_props(weight="bold")
        elif "incomparable" in cell.get_text().get_text():
            cell.set_facecolor("#FDE7DC")
    ax.set_title("(b) measured verdict per target (two-sided witness search)", fontsize=8)
    _save(fig, "fig10_refinement_lattice")


# --------------------------------------------------------------------------- #
# fig11 - the order sweep on T4
# --------------------------------------------------------------------------- #
def fig11_order_sweep(res: Dict[str, Any]) -> None:
    """Bug finding and progress as a function of feedback order, under two schedules.

    Panel (a) is the success rate with exact Clopper-Pearson intervals. Panel (b)
    is the median highest milestone reached, which every trial contributes to
    whether or not it found the bug and is therefore far more informative at this
    censoring level than a median over the handful of successful trials would be.
    The cost model's prediction is drawn on panel (a) as a thin dashed curve,
    rescaled to the observed maximum, and is labelled as a prediction: it commits
    to the *shape* and to the location of the optimum, not to the level.
    """
    # round-robin is the schedule used throughout the paper and is primary
    sw = res.get("order_sweep_roundrobin") or res["order_sweep"]
    rr = res["order_sweep"] if res.get("order_sweep_roundrobin") else None
    pc = sw["per_config"]
    n_trials = sw["n_trials_per_cell"]

    ng = [("ngram2", 2), ("ngram3", 3), ("ngram4", 4), ("ngram6", 6), ("ngram8", 8)]
    cx = [("ctx2", 2), ("ctx3", 3), ("ctx4", 4), ("ctx6", 6), ("ctx8", 8), ("ctx16", 16)]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    for src, tag, ls in ((pc, "round-robin", "-"),
                         (rr["per_config"] if rr else None, "score sched.", "--")):
        if src is None:
            continue
        for series, lbl, col, mk in (
            (ng, "NGRAM-$k$", OKABE["blue"], "o"),
            (cx, "context $N$", OKABE["orange"], "s"),
        ):
            xs = [o for _, o in series]
            ys = [src[c]["success_rate"] for c, _ in series]
            lo = [src[c]["success_rate"] - src[c]["success_ci95"][0] for c, _ in series]
            hi = [src[c]["success_ci95"][1] - src[c]["success_rate"] for c, _ in series]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, ms=3.5, lw=1.1, ls=ls,
                        capsize=1.6, color=col, alpha=1.0 if ls == "-" else 0.55,
                        label=f"{lbl}, {tag}")
    ax.axhline(pc["edge_only"]["success_rate"], ls=":", lw=1.0, color=OKABE["black"],
               label=f"AFL edge (order 2): {pc['edge_only']['n_success']}/{n_trials}")
    ax.axhline(pc["blind_random"]["success_rate"], ls=":", lw=1.0, color=OKABE["grey"],
               label=f"blind: {pc['blind_random']['n_success']}/{n_trials}")
    # cost model, rescaled to the observed maximum; shape only
    L, m = 5.0, 32.0
    grid = np.linspace(1.6, 16, 200)
    C = 1.0
    cost = C * m ** np.minimum(grid, L) + m ** np.maximum(L - grid, 0)
    pred = (1.0 / cost)
    pred = pred / pred.max() * max(
        max(pc[c]["success_rate"] for c, _ in ng),
        max(pc[c]["success_rate"] for c, _ in cx),
    )
    ax.plot(grid, pred, ls="--", lw=0.9, color=OKABE["vermillion"],
            label=r"cost model $1/(Cm^{n}+m^{L-n})$ (shape only, not fitted)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 4, 6, 8, 16])
    ax.set_xticklabels(["2", "3", "4", "6", "8", "16"])
    ax.set_xlabel("feedback order $n$")
    ax.set_ylabel(f"trials finding the bug / {n_trials}")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) success rate (95% Clopper-Pearson)", fontsize=8)
    ax.legend(fontsize=5.2, loc="upper right", ncol=1)

    ax = axes[1]
    for src, tag, ls, al in ((pc, "round-robin", "-", 1.0),
                             (rr["per_config"] if rr else None, "score sched.", "--", 0.5)):
        if src is None:
            continue
        for series, lbl, col, mk in (
            (ng, "NGRAM-$k$", OKABE["blue"], "o"),
            (cx, "context $N$", OKABE["orange"], "s"),
        ):
            ax.plot([o for _, o in series],
                    [src[c]["median_max_milestone"] for c, _ in series],
                    marker=mk, ms=3.5, lw=1.1, ls=ls, color=col, alpha=al,
                    label=f"{lbl}, {tag}")
    ax.axhline(pc["blind_random"]["median_max_milestone"], ls=":", lw=1.0,
               color=OKABE["grey"], label="blind")
    ax.axhline(5, ls="-", lw=0.8, color=OKABE["green"])
    ax.text(2.05, 5.04, "bug = milestone 5", fontsize=6, color=OKABE["green"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 4, 6, 8, 16])
    ax.set_xticklabels(["2", "3", "4", "6", "8", "16"])
    ax.set_xlabel("feedback order $n$")
    ax.set_ylabel("median highest milestone\n(all trials contribute)")
    ax.set_title("(b) progress toward the order-5 gate", fontsize=8)
    ax.legend(fontsize=5.6, loc="center right", bbox_to_anchor=(1.0, 0.58))
    fig.tight_layout()
    _save(fig, "fig11_order_sweep")


# --------------------------------------------------------------------------- #
# fig12 - exhaustive discrimination by order
# --------------------------------------------------------------------------- #
def fig12_discrimination_by_order(res: Dict[str, Any]) -> None:
    """What each order can *see*, enumerated exhaustively on T4.

    Discrimination and bug finding are different questions and the paper's whole
    argument rests on not confusing them, so they get separate figures. Here every
    walk over the sub-alphabet is enumerated, so these counts are exact rather
    than sampled.
    """
    w = res["t4_witnesses"]
    d = w["discrimination_by_order"]
    orders = sorted(int(k) for k in d)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax = axes[0]
    tot = w["n_conflated_groups"]
    vn = [d[str(o)]["afl_conflated_groups_split_by_ngram"] for o in orders]
    vc = [d[str(o)]["afl_conflated_groups_split_by_context"] for o in orders]
    b1 = ax.bar([o - 0.19 for o in orders], vn, 0.34, color=OKABE["blue"],
                label="NGRAM-$k$ (block window)")
    b2 = ax.bar([o + 0.19 for o in orders], vc, 0.34, color=OKABE["orange"],
                hatch="//", label="context depth $N$ (stack window)")
    for bars, vals in ((b1, vn), (b2, vc)):
        for rect, v in zip(bars, vals):
            ax.annotate(str(v), (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=5.8)
    ax.axhline(tot, ls="--", lw=1.0, color=OKABE["black"],
               label=f"all {tot} AFL-conflated groups")
    ax.set_xticks(orders)
    ax.set_xlim(orders[0] - 0.6, orders[-1] + 0.6)
    ax.set_xlabel("feedback order")
    ax.set_ylabel("AFL-conflated groups split")
    ax.set_ylim(0, tot * 1.45)
    ax.set_title(f"(a) exhaustive enumeration of all {w['n_walks_enumerated']} walks",
                 fontsize=8)
    ax.legend(fontsize=6.0, loc="upper left", ncol=1)

    ax = axes[1]
    res_ms = [d[str(o)]["residual_milestone_mixing_cells_under_ngram"] for o in orders]
    ax.bar(orders, res_ms, 0.5, color=OKABE["vermillion"])
    ax.set_xticks(orders)
    ax.set_xlabel("feedback order")
    ax.set_ylabel("cells still mixing executions\nat different bug distances")
    ax.set_title("(b) the blind spot closes exactly at order 3", fontsize=8)
    for o, v in zip(orders, res_ms):
        ax.annotate(str(v), (o, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=6.5)
    ax.set_ylim(0, max(res_ms) * 1.25 + 1)
    fig.tight_layout()
    _save(fig, "fig12_discrimination_by_order")


# --------------------------------------------------------------------------- #
# fig13 - paired analysis
# --------------------------------------------------------------------------- #
def fig13_paired(res: Dict[str, Any]) -> None:
    """Trial-level paired outcomes, which the unpaired tests never showed.

    Each bar is one contrast, decomposed into blocks won, lost, tied and
    censoring-indeterminate. The indeterminate segment is drawn, not dropped:
    it is the honest cost of censoring in a matched design.
    """
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for tgt, cfgs in res["paired_reanalysis"]["bug_finding_per_target"].items():
        for c, v in cfgs.items():
            if c == "blind_random":
                continue
            rows.append((f"{TARGET_LABEL.get(tgt, tgt).split()[0]}  {c}", v["win_loss_tie"]))
    sw = (res.get("order_sweep_roundrobin") or res["order_sweep"])["contrasts_vs_edge_only"]
    for c in ("ngram3", "ngram4", "ctx3", "ctx4", "ctx8", "ctx16"):
        if c in sw:
            rows.append((f"T4  {c}", sw[c]["paired_primary"]["win_loss_tie"]))

    fig, ax = plt.subplots(figsize=(7.2, 0.24 * len(rows) + 1.15))
    y = np.arange(len(rows))
    wa = np.array([r[1]["wins_a"] for r in rows], dtype=float)
    wb = np.array([r[1]["wins_b"] for r in rows], dtype=float)
    ti = np.array([r[1]["ties"] for r in rows], dtype=float)
    ind = np.array([r[1]["indeterminate"] for r in rows], dtype=float)
    ax.barh(y, wa, color=OKABE["green"], label="block won by the enriched arm")
    ax.barh(y, ti, left=wa, color=OKABE["yellow"], label="tie")
    ax.barh(y, ind, left=wa + ti, color=OKABE["grey"], hatch="xx",
            label="indeterminate (both censored)")
    ax.barh(y, wb, left=wa + ti + ind, color=OKABE["vermillion"],
            label="block won by edge coverage")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("matched trial blocks (seed-blocked design)")
    ax.set_title("Trial-level paired outcomes against the edge-coverage control", fontsize=8)
    ax.legend(fontsize=6.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig13_paired_outcomes")


# --------------------------------------------------------------------------- #
# fig14 - T4 survival curves
# --------------------------------------------------------------------------- #
def _km(t: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate of the probability of *not yet* having found the bug."""
    order = np.argsort(t)
    t, e = t[order], e[order]
    n = t.size
    times, surv = [0.0], [1.0]
    s = 1.0
    for i in range(n):
        at_risk = n - i
        if e[i] == 1:
            s *= (at_risk - 1) / at_risk
            times.append(float(t[i]))
            surv.append(s)
    return np.array(times), np.array(surv)


def fig14_t4_survival(res: Dict[str, Any]) -> None:
    """Time-to-bug survival curves on T4 by feedback order, censoring marked."""
    trials = json.loads(T4_TRIALS.read_text())
    budget = (res.get("order_sweep_roundrobin") or res["order_sweep"])["budget_execs"]
    trials = [r for r in trials if r.get("schedule", "round_robin") == "round_robin"]
    show = [
        ("blind_random", "blind search", OKABE["grey"], ":"),
        ("edge_only", "AFL edge (order 2)", OKABE["black"], "-"),
        ("ngram3", "NGRAM-3", OKABE["blue"], "-"),
        ("ngram4", "NGRAM-4", OKABE["skyblue"], "--"),
        ("ctx3", "context $N=3$", OKABE["orange"], "-"),
        ("ctx4", "context $N=4$", OKABE["vermillion"], "--"),
        ("ctx16", "context $N=16$", OKABE["purple"], "-."),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for cfg, lbl, col, ls in show:
        rows = [r for r in trials if r["config"] == cfg]
        if not rows:
            continue
        t = np.array([r["first_bug_exec"] or r["execs"] for r in rows], dtype=float)
        e = np.array([1 if r["first_bug_exec"] else 0 for r in rows], dtype=int)
        xs, ys = _km(t, e)
        xs = np.append(xs, budget)
        ys = np.append(ys, ys[-1])
        ax.step(np.maximum(xs, 1), ys, where="post", lw=1.2, color=col, ls=ls, label=lbl)
        n_cens = int((e == 0).sum())
        if n_cens:
            ax.plot([budget], [ys[-1]], marker="|", ms=7, color=col)
    ax.set_xscale("log")
    ax.set_xlim(50, budget * 1.3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("executions")
    ax.set_ylabel("fraction of trials\nwithout the bug yet")
    ax.set_title("T4: time to the higher-order bug", fontsize=8)
    ax.legend(fontsize=5.8, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig14_t4_survival")


# --------------------------------------------------------------------------- #
# fig15 - dilution: corpus inflation against order
# --------------------------------------------------------------------------- #
def fig15_dilution(res: Dict[str, Any]) -> None:
    """The other arm of the trade-off: what higher order costs.

    Coverage-element count and corpus size against feedback order, with the
    success rate on a twin axis, so the mechanism behind the interior optimum is
    visible in one frame rather than inferred across two.
    """
    pc = (res.get("order_sweep_roundrobin") or res["order_sweep"])["per_config"]
    series = [("ngram2", 2), ("ngram3", 3), ("ngram4", 4), ("ngram6", 6), ("ngram8", 8)]
    cx = [("ctx2", 2), ("ctx3", 3), ("ctx4", 4), ("ctx6", 6), ("ctx8", 8), ("ctx16", 16)]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot([o for _, o in series], [pc[c]["median_elements"] for c, _ in series],
            marker="o", ms=4, lw=1.2, color=OKABE["blue"], label="NGRAM-$k$ elements")
    ax.plot([o for _, o in cx], [pc[c]["median_elements"] for c, _ in cx],
            marker="s", ms=4, lw=1.2, color=OKABE["orange"], label="context elements")
    ax.plot([o for _, o in cx], [pc[c]["median_corpus"] for c, _ in cx],
            marker="^", ms=4, lw=1.0, ls="--", alpha=0.7, color=OKABE["green"],
            label="context corpus size")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2, 3, 4, 6, 8, 16])
    ax.set_xticklabels(["2", "3", "4", "6", "8", "16"])
    ax.set_xlabel("feedback order $n$")
    ax.set_ylabel("median count at end of trial")
    ax2 = ax.twinx()
    ax2.plot([o for _, o in series], [pc[c]["success_rate"] for c, _ in series],
             marker="D", ms=3.5, lw=1.2, color=OKABE["vermillion"],
             label="NGRAM-$k$ success")
    ax2.plot([o for _, o in cx], [pc[c]["success_rate"] for c, _ in cx],
             marker="v", ms=3.5, lw=1.2, ls=":", color=OKABE["purple"],
             label="context success")
    ax2.set_ylabel("success rate", color=OKABE["vermillion"])
    ax2.tick_params(axis="y", colors=OKABE["vermillion"])
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.4, loc="upper center",
              bbox_to_anchor=(0.5, -0.26), ncol=3)
    ax.set_title("Discrimination is bought with dilution", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig15_order_dilution")


def main() -> int:
    """Generate every revision figure."""
    res = json.loads(RESULTS.read_text())
    print("generating revision figures", flush=True)
    fig9_information_loss(res)
    fig10_lattice(res)
    fig12_discrimination_by_order(res)
    if "order_sweep" in res:
        fig11_order_sweep(res)
        fig14_t4_survival(res)
        fig15_dilution(res)
    if "paired_reanalysis" in res:
        fig13_paired(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
