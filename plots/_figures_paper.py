"""Consolidated multi-panel figures for the FSE 2027 submission.

Every number plotted here is read from the campaign result files; nothing is
typed in by hand.  Five vector figures are produced, each consolidating several
figures of the extended manuscript, all sized for the ACM ``acmsmall`` text
width (395.8pt = 5.49in):

``fig1_factorisation.pdf``
    (a) the four-stage factorisation of the AFL signal with the loss mechanism
    attached to each arrow; (b) the refinement order over the four dimensions.
``fig2_infoloss.pdf``
    (a) fraction of distinctions destroyed by each mechanism; (b) index
    collision as a function of map size.
``fig3_discrimination.pdf``
    exhaustive enumeration on T4: classes induced, conflated groups split and
    residual distance-mixing cells as a function of feedback order.
``fig4_ordersweep.pdf``
    (a) bug finding against feedback order; (b) corpus and element-space growth;
    (c) the cost model at three cost-growth exponents.
``fig5_capablation.pdf``
    the corpus-capped ablation: success rate with the corpus capacity pinned at
    the baseline's own corpus size, and success against realised corpus size.

Usage::

    python make_figures.py
"""

from __future__ import annotations

# --- artifact-evaluation path bootstrap (replaces the session-root anchor) --- #
import sys as _ae_sys  # noqa: E402
from pathlib import Path as _AEPath  # noqa: E402

AE_ROOT = _AEPath(__file__).resolve().parent.parent
if str(AE_ROOT) not in _ae_sys.path:
    _ae_sys.path.insert(0, str(AE_ROOT))
import ae_paths as _ae_paths  # noqa: E402,F401  (registers src/ and targets/)

AE_ROOT = AE_ROOT
# --- end bootstrap --------------------------------------------------------- #

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = _AEPath(__file__).resolve().parent.parent
SESSION = HERE
FIGS = HERE / "figures" / "paper"
FIGS.mkdir(parents=True, exist_ok=True)

REV = json.loads((SESSION / "results" / "step2_revision.json").read_text())
CAP_PATH = HERE / "results" / "corpus_cap_ablation.json"
W = 5.4  # inches: the acmsmall text width

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "lines.linewidth": 1.1,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C_EDGE = "#2B5D8A"
C_NGRAM = "#C1442E"
C_CTX = "#4E8A5C"
C_GREY = "#6B6B6B"
C_ACC = "#B8860B"
TARGETS = ["T1_context", "T2_value_range", "T3_state_machine", "T4_trigram"]
TLAB = ["T1", "T2", "T3", "T4"]


def panel_tag(ax, tag, dx=-0.16, dy=1.12):
    ax.text(
        dx, dy, tag, transform=ax.transAxes, fontweight="bold", fontsize=7.6,
        va="top", ha="left",
    )


# --------------------------------------------------------------------------- #
# Figure 1: factorisation chain (vertical) + refinement order
# --------------------------------------------------------------------------- #
def fig_factorisation(return_fig: bool = False):
    # ---------------- panel (a): vertical budget, in typographic points ------
    # The chain is laid out in points rather than in arbitrary data units: the
    # axes below is given a coordinate system in which one data unit is exactly
    # one point, so box heights, gap heights and font sizes are directly
    # comparable and the gap centres can be computed exactly.  The previous
    # revision placed the arrow labels at ``(y_top + y_bot + bh) / 2``, which
    # adds a spurious box height and pushes every annotation up into the box
    # above it; the midpoint is taken strictly over the gap here.
    TAG_PAD = 13.0    # headroom above the first box for the "(a)" tag
    BOX_H = 22.0      # label line (7.6pt) + subtitle line (5.5pt) + padding
    GAP_H = 20.0      # clear run between boxes: arrow, label and loss note
    BOX_PAD = 1.5     # outward padding added by the FancyBboxPatch boxstyle
    CAP_GAP = 6.0     # space under the last box
    CAP_H = 11.0      # the "map_0 = ..." caption line
    LAB_DY = 14.0     # box-label centre, measured up from the box bottom
    SUB_DY = 5.0      # box-subtitle centre, measured up from the box bottom
    N_BOX = 5
    PANEL_H = TAG_PAD + N_BOX * BOX_H + (N_BOX - 1) * GAP_H + CAP_GAP + CAP_H
    PANEL_B_H = 180.0  # the lattice panel, top-aligned with (a)

    GS_TOP, GS_BOT = 0.99, 0.01
    fig_h = PANEL_H / 72.0 / (GS_TOP - GS_BOT)
    fig = plt.figure(figsize=(W, fig_h))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.16, 1.0], height_ratios=[PANEL_B_H, PANEL_H - PANEL_B_H],
        wspace=0.06, hspace=0.0,
        left=0.012, right=0.988, top=GS_TOP, bottom=GS_BOT,
    )

    ax = fig.add_subplot(gs[:, 0])
    _p = ax.get_position()
    aw, ah = _p.width * W * 72.0, _p.height * fig_h * 72.0
    ax.set_xlim(0, aw)      # 1 data unit == 1 point, isotropically
    ax.set_ylim(0, ah)
    ax.axis("off")
    panel_tag(ax, "(a)", dx=0.0, dy=1.0)

    boxes = [
        (r"$e$", "execution"),
        (r"$\sigma(e)$", "block sequence"),
        (r"$G_2(\sigma)$", "bigram spectrum"),
        (r"$h(G_2)$", "bitmap counters"),
        (r"$q(r(h))$", "classified map"),
    ]
    x0, bw = 3.0, 78.0
    xm = x0 + bw / 2.0
    # top edge of box k; the bottom edge is tops[k] - BOX_H
    tops = [ah - TAG_PAD - k * (BOX_H + GAP_H) for k in range(N_BOX)]
    for (lab, sub), y_top in zip(boxes, tops):
        y_bot = y_top - BOX_H
        ax.add_patch(
            FancyBboxPatch(
                (x0, y_bot), bw, BOX_H,
                boxstyle=f"round,pad={BOX_PAD},rounding_size=3.0",
                linewidth=0.8, edgecolor=C_EDGE, facecolor="#EAF1F7",
            )
        )
        ax.text(xm, y_bot + LAB_DY, lab, ha="center", va="center", fontsize=7.6)
        ax.text(xm, y_bot + SUB_DY, sub, ha="center", va="center",
                fontsize=5.5, color=C_GREY)

    # one entry per inter-box gap, top to bottom
    losses = [
        (r"$\sigma$", None, None),
        (r"$G_2$", "higher-order sequence loss", C_NGRAM),
        (r"$h$", "index collision", C_ACC),
        (r"$q \circ r$", "counter representation,\nhit-count quantisation", C_CTX),
    ]
    for k, (lab, loss, col) in enumerate(losses):
        y_gap_top = tops[k] - BOX_H - BOX_PAD   # visible bottom of the upper box
        y_gap_bot = tops[k + 1] + BOX_PAD       # visible top of the lower box
        y_mid = (y_gap_top + y_gap_bot) / 2.0   # strict midpoint of the gap
        ax.add_patch(
            FancyArrowPatch((xm, y_gap_top), (xm, y_gap_bot), arrowstyle="-|>",
                            mutation_scale=7, linewidth=0.9, color="black")
        )
        ax.text(xm - 5.0, y_mid, lab, ha="right", va="center", fontsize=7.0)
        if loss:
            ax.plot([xm + 5.0, x0 + bw + 14.0], [y_mid, y_mid],
                    color=col, lw=0.7, ls=":")
            ax.text(x0 + bw + 18.0, y_mid, loss, ha="left", va="center",
                    fontsize=5.9, color=col, linespacing=1.15)
    ax.text(xm, CAP_H / 2.0,
            r"$\mathrm{map}_0 = q \circ r \circ h \circ G_2 \circ \sigma$",
            ha="center", va="center", fontsize=8.0)

    # ---------------- refinement order --------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-0.12, 1.12)
    ax2.set_ylim(-0.22, 1.16)
    ax2.axis("off")
    panel_tag(ax2, "(b)", dx=0.0, dy=1.0)

    nodes = {
        r"$\alpha_{\mathrm{3D}}$ composite": (0.5, 1.00),
        r"$\alpha_1$ context": (0.14, 0.62),
        r"$\alpha_2$ value": (0.52, 0.58),
        r"$\alpha_3$ state": (0.90, 0.62),
        r"$\alpha_0$ edge": (0.14, 0.20),
        r"$\bot$ blind": (0.5, -0.10),
    }
    edges = [
        (r"$\alpha_{\mathrm{3D}}$ composite", r"$\alpha_1$ context"),
        (r"$\alpha_{\mathrm{3D}}$ composite", r"$\alpha_2$ value"),
        (r"$\alpha_{\mathrm{3D}}$ composite", r"$\alpha_3$ state"),
        (r"$\alpha_1$ context", r"$\alpha_0$ edge"),
        (r"$\alpha_0$ edge", r"$\bot$ blind"),
        (r"$\alpha_2$ value", r"$\bot$ blind"),
        (r"$\alpha_3$ state", r"$\bot$ blind"),
    ]
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax2.plot([xa, xb], [ya, yb], color=C_GREY, lw=0.8, zorder=1)
    for lab, (x, y) in nodes.items():
        col = C_EDGE if "edge" in lab else ("#8C8C8C" if "blind" in lab else C_CTX)
        ax2.scatter([x], [y], s=15, color=col, zorder=3)
        ax2.text(x, y + 0.05, lab, ha="center", fontsize=6.4, zorder=4)
    ax2.annotate("", xy=(0.84, 0.50), xytext=(0.22, 0.27),
                 arrowprops=dict(arrowstyle="<->", color=C_NGRAM, lw=0.8,
                                 ls=(0, (3, 2))))
    # the dashed arrow runs under the label; a flush white patch keeps the
    # word legible without hiding either arrowhead
    ax2.text(0.53, 0.355, "incomparable", color=C_NGRAM, fontsize=6.4, ha="center",
             va="center", zorder=5,
             bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                       edgecolor="none"))
    ax2.text(0.5, -0.20, "edges run downward, finer to coarser", ha="center",
             fontsize=6.0, color=C_GREY)
    fig.savefig(FIGS / "fig1_factorisation.pdf")
    fig.savefig(FIGS / "fig1_factorisation.png", dpi=400)
    if return_fig:
        return fig
    plt.close(fig)
    return None


# --------------------------------------------------------------------------- #
# Figure 2: information loss
# --------------------------------------------------------------------------- #
def fig_infoloss() -> None:
    il = REV["information_loss"]["per_target"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.46))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    panel_tag(ax, "(a)")
    w = 0.26
    xs = np.arange(4)
    ho = [100 * il[t]["loss_higher_order_sequence"] for t in TARGETS]
    qz = [100 * il[t]["loss_hitcount_quantisation"] for t in TARGETS]
    co = [100 * max(il[t]["loss_index_collision"]["16"], 0.0) for t in TARGETS]
    ax.bar(xs - w, ho, w, color=C_NGRAM, label="higher-order")
    ax.bar(xs, qz, w, color=C_CTX, label="quantisation")
    ax.bar(xs + w, co, w, color=C_ACC, label=r"collision, $b{=}16$")
    for x, v in zip(xs - w, ho):
        if v > 0.3:
            ax.text(x, v + 0.4, f"{v:.1f}", ha="center", fontsize=5.6, color=C_NGRAM)
    for x, v in zip(xs, qz):
        if v > 0.3:
            ax.text(x, v + 0.4, f"{v:.1f}", ha="center", fontsize=5.6, color=C_CTX)
    ax.set_xticks(xs)
    ax.set_xticklabels(TLAB)
    ax.set_ylim(0, 20.5)
    ax.set_ylabel("distinctions destroyed (\\%)")
    ax.set_title("loss by mechanism", loc="left", fontsize=7.0)
    ax.legend(frameon=False, handlelength=1.0, labelspacing=0.25)

    ax = axes[1]
    panel_tag(ax, "(b)")
    bits = [8, 10, 12, 16]
    for t, lab, col, mk in zip(TARGETS, TLAB, [C_EDGE, C_ACC, C_CTX, C_NGRAM],
                               ["o", "s", "^", "D"]):
        base = il[t]["n_distinct_classified_maps_collision_free"]
        ys = [100.0 * (1.0 - il[t]["n_distinct_bitmap_states"][str(b)] / base)
              for b in bits]
        ax.plot(bits, ys, marker=mk, ms=2.8, color=col, label=lab)
    ctx = il["T1_context"]["context_dimension"]
    ax.plot(bits, [100 * ctx["loss_index_collision"][str(b)] for b in bits],
            marker="v", ms=2.8, ls="--", color=C_GREY, label=r"T1, $\alpha_1$")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks(bits)
    ax.set_xlabel("map size exponent $b$")
    ax.set_ylabel("collision loss (\\%)")
    ax.set_title("collision vs map size", loc="left", fontsize=7.0)
    ax.legend(frameon=False, ncol=2, handlelength=1.1, loc="upper right",
              labelspacing=0.25, columnspacing=0.9)
    fig.savefig(FIGS / "fig2_infoloss.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: discrimination by order (exhaustive enumeration on T4)
# --------------------------------------------------------------------------- #
def fig_discrimination() -> None:
    t4 = REV["t4_witnesses"]
    dbo = t4["discrimination_by_order"]
    orders = sorted(int(k) for k in dbo)
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.46))
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    panel_tag(ax, "(a)")
    ax.plot(orders, [dbo[str(o)]["ngram_classes"] for o in orders], marker="o",
            ms=2.8, color=C_NGRAM, label=r"NGRAM-$k$")
    ax.plot(orders, [dbo[str(o)]["context_classes"] for o in orders], marker="s",
            ms=2.8, color=C_CTX, label=r"context-$N$")
    ax.axhline(t4["n_d0_counted_classes"], color=C_EDGE, ls="--", lw=0.8)
    ax.text(6.0, t4["n_d0_counted_classes"] - 58, "edge coverage", color=C_EDGE,
            fontsize=5.8, ha="right")
    ax.axhline(t4["n_walks_enumerated"], color="black", ls=":", lw=0.7)
    ax.text(6.0, t4["n_walks_enumerated"] + 8, "all 729 walks", fontsize=5.8,
            ha="right")
    ax.set_xlabel("feedback order")
    ax.set_ylabel("classes induced")
    ax.set_ylim(330, 800)
    ax.set_xticks(orders)
    ax.legend(frameon=False, loc="center right", handlelength=1.1, labelspacing=0.25)

    ax = axes[1]
    panel_tag(ax, "(b)")
    n_conf = t4["n_conflated_groups"]
    ax.bar(np.array(orders) - 0.16,
           [dbo[str(o)]["afl_conflated_groups_split_by_ngram"] for o in orders],
           0.34, color=C_NGRAM, label="groups split")
    ax.axhline(n_conf, color=C_EDGE, ls="--", lw=0.8)
    ax.text(2.0, n_conf + 7, f"all {n_conf} conflated groups", fontsize=5.6,
            ha="left", color=C_EDGE)
    ax.text(1.84, 7, "0", fontsize=5.8, ha="center", color=C_NGRAM)
    ax.set_ylim(0, 250)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(orders,
             [dbo[str(o)]["residual_milestone_mixing_cells_under_ngram"] for o in orders],
             marker="D", ms=2.8, color=C_ACC, label="residual mixed cells")
    ax2.set_ylabel("residual mixed cells", color=C_ACC, fontsize=6.4)
    ax2.tick_params(axis="y", colors=C_ACC, labelsize=6.0)
    ax.set_xlabel("feedback order")
    ax.set_ylabel("conflated groups split")
    ax.set_xticks(orders)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc=(0.20, 0.40), handlelength=1.1,
              labelspacing=0.25)
    fig.savefig(FIGS / "fig3_discrimination.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: order sweep, corpus growth and the cost model
# --------------------------------------------------------------------------- #
def fig_ordersweep() -> float:
    sw = REV["order_sweep_roundrobin"]["per_config"]
    ng = [("ngram2", 2), ("ngram3", 3), ("ngram4", 4), ("ngram6", 6), ("ngram8", 8)]
    ct = [("ctx2", 2), ("ctx3", 3), ("ctx4", 4), ("ctx6", 6), ("ctx8", 8), ("ctx16", 16)]

    fig, axes = plt.subplots(1, 3, figsize=(W, 1.46))
    fig.subplots_adjust(wspace=0.46)

    ax = axes[0]
    panel_tag(ax, "(a)", dx=-0.30)
    for series, col, mk, lab in ((ng, C_NGRAM, "o", r"NGRAM-$k$"),
                                 (ct, C_CTX, "s", r"context-$N$")):
        xs = [o for _, o in series]
        ys = [100 * sw[c]["success_rate"] for c, _ in series]
        lo = [100 * sw[c]["success_ci95"][0] for c, _ in series]
        hi = [100 * sw[c]["success_ci95"][1] for c, _ in series]
        ax.plot(xs, ys, marker=mk, ms=2.8, color=col, label=lab)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
    ax.axhline(100 * sw["edge_only"]["success_rate"], color=C_EDGE, ls="--", lw=0.8)
    ax.text(16, 100 * sw["edge_only"]["success_rate"] + 3, "edge", color=C_EDGE,
            fontsize=5.8, ha="right")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 4, 6, 8, 16])
    ax.set_xticklabels(["2", "3", "4", "6", "8", "16"])
    ax.set_xlabel("feedback order")
    ax.set_ylabel("bug found (\\% of trials)")
    ax.legend(frameon=False, handlelength=1.0, loc="upper right", labelspacing=0.25)

    ax = axes[1]
    panel_tag(ax, "(b)", dx=-0.32)
    for series, col, mk, lab in ((ng, C_NGRAM, "o", "corpus, NGRAM"),
                                 (ct, C_CTX, "s", "corpus, context")):
        ax.plot([o for _, o in series], [sw[c]["median_corpus"] for c, _ in series],
                marker=mk, ms=2.8, color=col, label=lab)
    ax.plot([o for _, o in ng], [sw[c]["median_elements"] for c, _ in ng], marker="^",
            ms=2.8, ls=":", color=C_GREY, label="elements")
    m = 32.0
    k2 = sw["ngram2"]["median_corpus"]
    k3 = sw["ngram3"]["median_corpus"]
    gamma = math.log(k3 / k2) / math.log(m)
    xs = np.linspace(2, 5.4, 40)
    ax.plot(xs, k2 * m ** (gamma * (xs - 2)), color=C_ACC, lw=0.9, ls="--",
            label=rf"fit $\hat\beta={gamma:.2f}$")
    ax.set_yscale("log")
    ax.set_ylim(6e2, 8e6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 4, 6, 8, 16])
    ax.set_xticklabels(["2", "3", "4", "6", "8", "16"])
    ax.set_xlabel("feedback order")
    ax.set_ylabel("median count")
    ax.legend(frameon=False, handlelength=1.0, loc="lower right", labelspacing=0.15,
              borderpad=0.1, fontsize=5.4)

    ax = axes[2]
    panel_tag(ax, "(c)", dx=-0.34)
    L, mm = 5.0, 32.0
    ns = np.linspace(1.0, 6.0, 501)
    for b, col, ls, tag in ((gamma, C_NGRAM, "-", r"$\beta=\hat\beta$"),
                            (1.0, C_CTX, "--", r"$\beta=1$"),
                            (2.0, C_EDGE, ":", r"$\beta=2$")):
        T = mm ** (b * ns) + mm ** (L - ns)
        ax.plot(ns, T, color=col, ls=ls, label=tag)
        j = int(np.argmin(T))
        ax.scatter([ns[j]], [T[j]], s=11, color=col, zorder=4)
    ax.axvline(3, color=C_GREY, lw=0.7, ls="-.")
    ax.text(2.90, 1.5e2, "observed", fontsize=5.7, color=C_GREY, rotation=90,
            va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_ylim(5e1, 1e9)
    ax.set_xlabel("feedback order $n$")
    ax.set_ylabel("modelled cost $T(n)$")
    ax.legend(frameon=False, handlelength=1.3, loc="upper center", labelspacing=0.15,
              fontsize=5.6, borderpad=0.1)
    fig.savefig(FIGS / "fig4_ordersweep.pdf")
    plt.close(fig)
    for b in (gamma, 1.0, 2.0):
        T = mm ** (b * ns) + mm ** (L - ns)
        print(f"beta={b:.3f} argmin={ns[int(np.argmin(T))]:.2f} L/(1+b)={L/(1+b):.2f}")
    print(f"gamma_hat={gamma:.4f} k2={k2:.0f} k3={k3:.0f}")
    return gamma


# --------------------------------------------------------------------------- #
# Figure 5: corpus-capped ablation
# --------------------------------------------------------------------------- #
def fig_capablation() -> None:
    if not CAP_PATH.exists():
        print("cap ablation results absent; skipping fig5")
        return
    cap = json.loads(CAP_PATH.read_text())
    pa = cap["per_arm"]
    n = cap["design"]["n_trials_per_arm"]
    orders = [3, 4, 6, 8]
    unc = [pa[f"ngram{o}"] for o in orders]
    cpd = [pa[f"ngram{o}_cap_edge"] for o in orders]

    fig, axes = plt.subplots(1, 2, figsize=(W, 1.46))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    panel_tag(ax, "(a)")
    xs = np.arange(len(orders))
    w = 0.34
    ax.bar(xs - w / 2, [100 * a["success_rate"] for a in unc], w, color=C_NGRAM,
           label="unbounded corpus")
    ax.bar(xs + w / 2, [100 * a["success_rate"] for a in cpd], w, color=C_EDGE,
           label=f"capped at {cap['design']['cap_levels']['edge']:,}")
    for x, a in zip(xs - w / 2, unc):
        ax.text(x, 100 * a["success_rate"] + 1.5, f"{a['n_success']}", ha="center",
                fontsize=5.6, color=C_NGRAM)
    for x, a in zip(xs + w / 2, cpd):
        ax.text(x, 100 * a["success_rate"] + 1.5, f"{a['n_success']}", ha="center",
                fontsize=5.6, color=C_EDGE)
    ax.axhline(100 * pa["edge_only"]["success_rate"], color=C_GREY, ls="--", lw=0.8)
    ax.text(len(orders) - 0.5, 100 * pa["edge_only"]["success_rate"] + 2.0,
            "edge coverage", fontsize=5.7, color=C_GREY, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"order {o}" for o in orders])
    ax.set_ylim(0, 118)
    ax.set_ylabel(f"bug found (\\% of {n} trials)")
    ax.legend(frameon=False, handlelength=1.0, loc="upper right", labelspacing=0.25)

    ax = axes[1]
    panel_tag(ax, "(b)")
    ax.plot([a["median_corpus"] for a in unc], [100 * a["success_rate"] for a in unc],
            marker="o", ms=3.0, ls="", color=C_NGRAM, label="unbounded")
    ax.plot([a["median_corpus"] for a in cpd], [100 * a["success_rate"] for a in cpd],
            marker="s", ms=3.0, ls="", color=C_EDGE, label="capped")
    for a, b, o in zip(unc, cpd, orders):
        ax.annotate("", xy=(b["median_corpus"], 100 * b["success_rate"]),
                    xytext=(a["median_corpus"], 100 * a["success_rate"]),
                    arrowprops=dict(arrowstyle="->", lw=0.6, color=C_GREY))
        ax.text(a["median_corpus"] * 1.10, 100 * a["success_rate"] + 1.5, f"{o}",
                fontsize=5.8, color=C_NGRAM)
    e = pa["edge_only"]
    ax.plot([e["median_corpus"]], [100 * e["success_rate"]], marker="*", ms=5.5,
            color=C_GREY, ls="", label="edge coverage")
    ax.set_xscale("log")
    ax.set_ylim(-6, 110)
    ax.set_xticks([2e3, 1e4, 4e4])
    ax.set_xticklabels(["$2{\\times}10^3$", "$10^4$", "$4{\\times}10^4$"])
    ax.minorticks_off()
    ax.set_xlabel("median corpus size")
    ax.set_ylabel("bug found (\\% of trials)")
    ax.legend(frameon=False, handlelength=1.0, loc="upper right", labelspacing=0.25)
    fig.savefig(FIGS / "fig5_capablation.pdf")
    plt.close(fig)
    for arm, a in pa.items():
        res = a.get("expected_residence_execs")
        print(f"{arm:>18s} {a['n_success']:>2d}/{n} corpus={a['median_corpus']:>9,.0f} "
              f"admitted={a.get('median_n_admitted', float('nan')):>10,.0f} "
              f"residence={(res if res is not None else float('nan')):>10,.0f}")


if __name__ == "__main__":
    fig_factorisation()
    fig_infoloss()
    fig_discrimination()
    fig_ordersweep()
    fig_capablation()
    print("figures written to", FIGS)
