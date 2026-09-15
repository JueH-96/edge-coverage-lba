"""
Statistical utilities for the fuzzing benchmark comparison.

The protocol follows Klees et al., "Evaluating Fuzz Testing" (CCS 2018): many
independent trials, a statistical test rather than a single run, a standardised
effect size, and explicit handling of trials that never trigger the bug
(right-censored observations). Multiple comparisons across the family of
(target, configuration) contrasts are corrected with Benjamini-Hochberg FDR.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "vargha_delaney_a12",
    "a12_magnitude",
    "mann_whitney",
    "wilcoxon_paired",
    "clopper_pearson",
    "fisher_success",
    "logrank_test",
    "bootstrap_ci",
    "benjamini_hochberg",
    "describe",
]


def vargha_delaney_a12(x: Sequence[float], y: Sequence[float]) -> float:
    """Vargha-Delaney A12 effect size.

    ``A12 = P(X > Y) + 0.5 * P(X == Y)``: the probability that a randomly drawn
    observation from ``x`` exceeds one from ``y``. 0.5 means no effect, 1.0 means
    ``x`` always exceeds ``y``.

    Parameters
    ----------
    x, y : sequence of float
        Independent samples.

    Returns
    -------
    float
        A12 in ``[0, 1]``; ``nan`` if either sample is empty.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # rank-based formulation, O(n log n)
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    r1 = float(np.sum(ranks[: a.size]))
    m, n = a.size, b.size
    return (r1 / m - (m + 1) / 2.0) / n


def a12_magnitude(a12: float) -> str:
    """Vargha-Delaney magnitude label for an A12 value.

    Thresholds on ``|A12 - 0.5|``: < 0.06 negligible, < 0.14 small, < 0.21 medium,
    otherwise large.

    Parameters
    ----------
    a12 : float
        Effect size.

    Returns
    -------
    str
        One of ``"negligible"``, ``"small"``, ``"medium"``, ``"large"``, ``"undefined"``.
    """
    if not math.isfinite(a12):
        return "undefined"
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def mann_whitney(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Two-sided Mann-Whitney U test with A12 effect size.

    Parameters
    ----------
    x, y : sequence of float
        Independent samples.

    Returns
    -------
    dict
        ``U``, ``p_value``, ``a12``, ``a12_magnitude``.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    out: Dict[str, float] = {}
    if a.size == 0 or b.size == 0:
        return {"U": float("nan"), "p_value": float("nan"), "a12": float("nan"),
                "a12_magnitude": "undefined"}
    # Identical constant samples make the test degenerate; report p = 1 explicitly.
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return {"U": float(a.size * b.size / 2), "p_value": 1.0, "a12": 0.5,
                "a12_magnitude": "negligible"}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    a12 = vargha_delaney_a12(a, b)
    out["U"] = float(u)
    out["p_value"] = float(p)
    out["a12"] = float(a12)
    out["a12_magnitude"] = a12_magnitude(a12)
    return out


def wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test for the blocked (paired-by-seed) design.

    Parameters
    ----------
    x, y : sequence of float
        Paired samples of equal length (trial ``i`` shares its seed across
        configurations).

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``n_pairs``, ``n_nonzero_pairs``.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size != b.size or a.size == 0:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n_pairs": int(min(a.size, b.size)), "n_nonzero_pairs": 0}
    d = a - b
    nz = int(np.count_nonzero(d))
    if nz == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": int(a.size),
                "n_nonzero_pairs": 0}
    try:
        st, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n_pairs": int(a.size), "n_nonzero_pairs": nz}
    return {"statistic": float(st), "p_value": float(p), "n_pairs": int(a.size),
            "n_nonzero_pairs": nz}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    alpha : float, optional
        Significance level, default 0.05 (95% CI).

    Returns
    -------
    tuple of float
        ``(lower, upper)`` bounds on the success probability.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def fisher_success(k1: int, n1: int, k2: int, n2: int) -> Dict[str, float]:
    """Fisher's exact test on two success counts.

    Parameters
    ----------
    k1, n1 : int
        Successes and trials in group 1.
    k2, n2 : int
        Successes and trials in group 2.

    Returns
    -------
    dict
        ``p_value``, per-group rates with 95% Clopper-Pearson CIs, and two odds
        ratios. The raw odds ratio is ``None`` when a zero cell makes it undefined
        (infinite, or indeterminate when both groups have zero successes), with
        ``odds_ratio_undefined_zero_cell`` set; ``odds_ratio_haldane_anscombe``
        applies the standard +0.5 continuity correction to every cell and is always
        finite, so it remains reportable for the 0/30-vs-24/30 style contrasts that
        dominate this benchmark.

    Notes
    -----
    The p-value is unaffected by the zero cell - Fisher's exact test is still valid -
    so significance is read from ``p_value`` and the corrected odds ratio is used
    only as an effect-size descriptor.
    """
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    orr, p = stats.fisher_exact(table, alternative="two-sided")
    lo1, hi1 = clopper_pearson(k1, n1)
    lo2, hi2 = clopper_pearson(k2, n2)

    orr = float(orr)
    undefined = not math.isfinite(orr)
    # Haldane-Anscombe: add 0.5 to every cell so the ratio is always defined.
    a, b = k1 + 0.5, (n1 - k1) + 0.5
    c, d = k2 + 0.5, (n2 - k2) + 0.5
    or_ha = (a / b) / (c / d)

    return {
        "odds_ratio": None if undefined else orr,
        "odds_ratio_undefined_zero_cell": undefined,
        "odds_ratio_haldane_anscombe": float(or_ha),
        "p_value": float(p),
        "rate_1": k1 / n1 if n1 else None,
        "rate_1_ci95": [lo1, hi1],
        "rate_2": k2 / n2 if n2 else None,
        "rate_2_ci95": [lo2, hi2],
    }


def logrank_test(
    t1: Sequence[float], e1: Sequence[int], t2: Sequence[float], e2: Sequence[int]
) -> Dict[str, float]:
    """Two-sample log-rank test for right-censored time-to-event data.

    This is the appropriate test for "executions until the bug is found" when some
    trials exhaust the budget without finding it: those observations are censored,
    not equal to the budget, and a rank test on budget-substituted values is only
    conservative rather than correct.

    Parameters
    ----------
    t1, t2 : sequence of float
        Observed times (execution counts) per trial.
    e1, e2 : sequence of int
        Event indicators: 1 = bug found at that time, 0 = right-censored.

    Returns
    -------
    dict
        ``chi2``, ``p_value``, ``observed_1``, ``expected_1``, ``n_events``.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    e1 = np.asarray(e1, dtype=int)
    e2 = np.asarray(e2, dtype=int)
    n_events = int(e1.sum() + e2.sum())
    if n_events == 0:
        return {"chi2": 0.0, "p_value": 1.0, "observed_1": 0.0, "expected_1": 0.0,
                "n_events": 0}

    times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    o1 = 0.0
    exp1 = 0.0
    var = 0.0
    for t in times:
        n1 = float(np.sum(t1 >= t))
        n2 = float(np.sum(t2 >= t))
        n = n1 + n2
        if n <= 1:
            continue
        d1 = float(np.sum((t1 == t) & (e1 == 1)))
        d2 = float(np.sum((t2 == t) & (e2 == 1)))
        d = d1 + d2
        if d == 0:
            continue
        o1 += d1
        exp1 += d * n1 / n
        var += d * (n - d) * n1 * n2 / (n * n * (n - 1))
    if var <= 0:
        return {"chi2": 0.0, "p_value": 1.0, "observed_1": o1, "expected_1": exp1,
                "n_events": n_events}
    chi2 = (o1 - exp1) ** 2 / var
    p = float(stats.chi2.sf(chi2, df=1))
    return {"chi2": float(chi2), "p_value": p, "observed_1": float(o1),
            "expected_1": float(exp1), "n_events": n_events}


def bootstrap_ci(
    x: Sequence[float],
    statistic: str = "median",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 20260804,
) -> Tuple[float, float, float]:
    """Percentile bootstrap confidence interval.

    Parameters
    ----------
    x : sequence of float
        Sample.
    statistic : {"median", "mean"}, optional
        Statistic to bootstrap, default ``"median"``.
    n_boot : int, optional
        Bootstrap resamples, default 10000.
    alpha : float, optional
        Significance level, default 0.05.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    tuple of float
        ``(point_estimate, lower, upper)``.
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    fn = np.median if statistic == "median" else np.mean
    point = float(fn(a))
    if a.size == 1:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = fn(a[idx], axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, list]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values (``nan`` entries are passed through as ``nan``).
    alpha : float, optional
        Target false-discovery rate, default 0.05.

    Returns
    -------
    dict
        ``q_values`` (adjusted p-values, same order as input), ``rejected`` (bool
        list), and ``alpha``.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    q = np.full(n, np.nan)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return {"q_values": q.tolist(), "rejected": [False] * n, "alpha": alpha}
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx], kind="stable")]
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    q[order] = adj
    rejected = [bool(np.isfinite(v) and v <= alpha) for v in q]
    return {"q_values": q.tolist(), "rejected": rejected, "alpha": alpha}


def describe(x: Sequence[float]) -> Dict[str, float]:
    """Summary statistics with a bootstrap median CI.

    Parameters
    ----------
    x : sequence of float
        Sample.

    Returns
    -------
    dict
        ``n``, ``mean``, ``sd``, ``min``, ``q1``, ``median``, ``q3``, ``max``,
        ``median_ci95``.
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {"n": 0}
    med, lo, hi = bootstrap_ci(a, "median")
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "min": float(np.min(a)),
        "q1": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "q3": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
        "median_ci95": [lo, hi],
    }
