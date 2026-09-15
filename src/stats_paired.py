"""
Paired (randomized-block) inference for the fuzzing benchmark comparison.

Why this module exists
----------------------
The campaign is a **randomized block design**: trial ``i`` of every configuration
is driven by the same derived seed, so the trials are matched in blocks rather
than independent between configurations. Fisher's exact test, the ordinary
two-sample log-rank test and the Vargha-Delaney statistic are all *unpaired*
procedures. Applied to a blocked design they remain valid tests of the null
hypothesis of no effect - the randomisation is still exchangeable - but they
throw away the block structure, and therefore power, that the design was built to
provide. Worse, they answer a marginal rather than a within-block question, so
they can disagree with the design's own logic when block-level variance is large.

This module supplies the matched-design counterparts that are primary in the
revised evaluation:

``mcnemar_exact``
    Exact conditional (binomial) McNemar test for paired binary success.
``paired_win_loss_tie``
    Censoring-aware within-block comparison of time-to-bug. A block is a *win*,
    a *loss*, or *indeterminate*; indeterminate blocks are exactly those in which
    censoring makes the ordering unknowable, and they are reported rather than
    silently coerced.
``sign_test_exact``
    Exact binomial sign test on the win/loss counts (the matched-pairs analogue
    of Gehan's generalised Wilcoxon).
``paired_permutation``
    Exact-when-small, Monte-Carlo-otherwise sign-flip permutation test on paired
    differences of log time-to-bug, restricted to blocks where both arms observed
    the event, with the discarded fraction reported.
``stratified_logrank``
    Log-rank test stratified on the seed block: the score and its variance are
    accumulated *within* strata and pooled, so between-block variation never
    enters the test statistic.
``paired_prob_superiority``
    Within-block probability of superiority with a bootstrap interval - the
    paired analogue of the A12 statistic.

Every function returns a plain dictionary of JSON-serialisable scalars so the
results file can be regenerated and diffed.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "mcnemar_exact",
    "paired_win_loss_tie",
    "sign_test_exact",
    "paired_permutation",
    "stratified_logrank",
    "paired_prob_superiority",
]


# --------------------------------------------------------------------------- #
# Paired binary outcome
# --------------------------------------------------------------------------- #
def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> Dict[str, float]:
    """Exact conditional McNemar test for paired binary outcomes.

    Conditional on the number of discordant blocks ``n_d = n_01 + n_10``, the
    count ``n_10`` is binomial ``(n_d, 1/2)`` under the null, so the exact
    two-sided p-value is the doubled tail probability truncated at 1. Concordant
    blocks carry no information about the treatment difference and are correctly
    excluded, which is the entire point of the paired analysis.

    Parameters
    ----------
    a, b : sequence of int
        Matched binary outcomes (1 = success) for the two configurations; ``a[i]``
        and ``b[i]`` must come from the same block.

    Returns
    -------
    dict
        ``n_pairs``, ``n_11``, ``n_10``, ``n_01``, ``n_00``, ``n_discordant``,
        ``p_value``, ``odds_ratio_cond`` (``n_10 / n_01``, ``inf`` when
        ``n_01 = 0``), and ``risk_difference`` (``(n_10 - n_01) / n_pairs``).

    Raises
    ------
    ValueError
        If the two sequences have different lengths.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("paired sequences must have equal length")
    n = int(a.size)
    n11 = int(np.sum((a == 1) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    nd = n10 + n01
    if nd == 0:
        p = 1.0
    else:
        k = min(n10, n01)
        p = float(min(1.0, 2.0 * stats.binom.cdf(k, nd, 0.5)))
    return {
        "test": "mcnemar_exact",
        "n_pairs": n,
        "n_11": n11,
        "n_10": n10,
        "n_01": n01,
        "n_00": n00,
        "n_discordant": nd,
        "p_value": p,
        "odds_ratio_cond": (float(n10) / n01) if n01 else (float("inf") if n10 else 1.0),
        "risk_difference": (n10 - n01) / n if n else 0.0,
    }


# --------------------------------------------------------------------------- #
# Paired time-to-event under right censoring
# --------------------------------------------------------------------------- #
def paired_win_loss_tie(
    ta: Sequence[float],
    ea: Sequence[int],
    tb: Sequence[float],
    eb: Sequence[int],
) -> Dict[str, float]:
    """Censoring-aware within-block comparison of two time-to-event arms.

    For block ``i`` the arm with the *smaller* time is better (fewer executions to
    the bug). Comparability under right censoring follows the standard matched
    win-ratio convention:

    * both observed  -> the smaller time wins, exact equality is a tie;
    * ``a`` observed, ``b`` censored -> ``a`` wins iff ``t_a <= t_b``, otherwise
      indeterminate (``b`` might still have found the bug just after its budget);
    * ``b`` observed, ``a`` censored -> symmetric;
    * both censored  -> indeterminate.

    Indeterminate blocks are counted and reported, never recoded as ties, because
    recoding them would bias the sign test toward the null in exactly the regime -
    heavy censoring - where the difference is largest.

    Parameters
    ----------
    ta, tb : sequence of float
        Observed or censoring times per block.
    ea, eb : sequence of int
        Event indicators (1 = bug found, 0 = right-censored).

    Returns
    -------
    dict
        ``n_pairs``, ``wins_a``, ``wins_b``, ``ties``, ``indeterminate``,
        ``win_ratio`` and ``win_fraction_a`` (over comparable, non-tied blocks).
    """
    ta = np.asarray(ta, dtype=float)
    tb = np.asarray(tb, dtype=float)
    ea = np.asarray(ea, dtype=int)
    eb = np.asarray(eb, dtype=int)
    wins_a = wins_b = ties = indet = 0
    for i in range(ta.size):
        if ea[i] and eb[i]:
            if ta[i] < tb[i]:
                wins_a += 1
            elif tb[i] < ta[i]:
                wins_b += 1
            else:
                ties += 1
        elif ea[i] and not eb[i]:
            if ta[i] <= tb[i]:
                wins_a += 1
            else:
                indet += 1
        elif eb[i] and not ea[i]:
            if tb[i] <= ta[i]:
                wins_b += 1
            else:
                indet += 1
        else:
            indet += 1
    comparable = wins_a + wins_b
    return {
        "n_pairs": int(ta.size),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "indeterminate": indet,
        "win_ratio": (float(wins_a) / wins_b) if wins_b else (float("inf") if wins_a else 1.0),
        "win_fraction_a": (wins_a / comparable) if comparable else 0.5,
    }


def sign_test_exact(wins: int, losses: int) -> Dict[str, float]:
    """Exact two-sided binomial sign test on win/loss counts.

    Parameters
    ----------
    wins, losses : int
        Numbers of blocks won by each arm (ties and indeterminate blocks are
        excluded by the caller).

    Returns
    -------
    dict
        ``n_effective``, ``p_value`` and the exact Clopper-Pearson interval for
        the win probability.
    """
    n = int(wins + losses)
    if n == 0:
        return {"test": "sign_exact", "n_effective": 0, "p_value": 1.0,
                "win_prob": 0.5, "win_prob_lo": 0.0, "win_prob_hi": 1.0}
    k = int(wins)
    p = float(min(1.0, 2.0 * stats.binom.cdf(min(k, n - k), n, 0.5)))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return {
        "test": "sign_exact",
        "n_effective": n,
        "p_value": p,
        "win_prob": k / n,
        "win_prob_lo": lo,
        "win_prob_hi": hi,
    }


def paired_permutation(
    ta: Sequence[float],
    ea: Sequence[int],
    tb: Sequence[float],
    eb: Sequence[int],
    n_perm: int = 100_000,
    seed: int = 20260804,
) -> Dict[str, float]:
    """Sign-flip permutation test on paired differences of log time-to-bug.

    The test statistic is the mean of ``log(t_a) - log(t_b)`` over blocks in which
    *both* arms observed the event; under the null of exchangeability within a
    block, each difference is equally likely to carry either sign, so the null
    distribution is generated by independent sign flips. With at most 20 usable
    blocks the full ``2**n`` enumeration is exact; above that a Monte-Carlo
    approximation with ``n_perm`` draws is used and reported as such.

    The number of blocks discarded because at least one arm was censored is
    returned as ``n_dropped``. When censoring is heavy this test is *not* the
    primary analysis - :func:`paired_win_loss_tie` with :func:`sign_test_exact`
    is - because conditioning on both arms observing the event is informative.

    Parameters
    ----------
    ta, tb : sequence of float
        Observed or censoring times per block.
    ea, eb : sequence of int
        Event indicators (1 = observed).
    n_perm : int, optional
        Monte-Carlo replicate count when exact enumeration is infeasible.
    seed : int, optional
        RNG seed for the Monte-Carlo branch.

    Returns
    -------
    dict
        ``n_used``, ``n_dropped``, ``mean_log_ratio``, ``median_ratio``,
        ``p_value``, ``exact`` (bool as int), ``n_perm``.
    """
    ta = np.asarray(ta, dtype=float)
    tb = np.asarray(tb, dtype=float)
    ea = np.asarray(ea, dtype=int)
    eb = np.asarray(eb, dtype=int)
    both = (ea == 1) & (eb == 1)
    d = np.log(np.maximum(ta[both], 1.0)) - np.log(np.maximum(tb[both], 1.0))
    n_used = int(d.size)
    n_dropped = int(ta.size - n_used)
    if n_used == 0:
        return {"test": "paired_permutation", "n_used": 0, "n_dropped": n_dropped,
                "mean_log_ratio": 0.0, "median_ratio": 1.0, "p_value": 1.0,
                "exact": 0, "n_perm": 0}
    obs = float(np.mean(d))
    if n_used <= 20:
        signs = np.array(
            [[1 if (m >> j) & 1 else -1 for j in range(n_used)] for m in range(1 << n_used)],
            dtype=float,
        )
        null = signs @ d / n_used
        p = float(np.mean(np.abs(null) >= abs(obs) - 1e-15))
        exact, npm = 1, int(1 << n_used)
    else:
        rng = np.random.default_rng(seed)
        flips = rng.integers(0, 2, size=(n_perm, n_used)) * 2 - 1
        null = flips @ d / n_used
        p = float((np.sum(np.abs(null) >= abs(obs) - 1e-15) + 1) / (n_perm + 1))
        exact, npm = 0, int(n_perm)
    return {
        "test": "paired_permutation",
        "n_used": n_used,
        "n_dropped": n_dropped,
        "mean_log_ratio": obs,
        "median_ratio": float(np.exp(np.median(d))),
        "p_value": p,
        "exact": exact,
        "n_perm": npm,
    }


def stratified_logrank(
    ta: Sequence[float],
    ea: Sequence[int],
    tb: Sequence[float],
    eb: Sequence[int],
    strata: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Log-rank test stratified on the trial block.

    The score ``O - E`` and its hypergeometric variance are accumulated *within*
    each stratum and summed, so the statistic is

        chi2 = (sum_s (O_s - E_s))**2 / sum_s V_s

    with one degree of freedom. With one matched pair per stratum this is the
    stratified analogue of the sign test and, unlike the pooled log-rank test,
    contains no contribution from between-block variation in difficulty.

    Parameters
    ----------
    ta, tb : sequence of float
        Times per trial, ordered so that index ``i`` is block ``i`` in both arms.
    ea, eb : sequence of int
        Event indicators.
    strata : sequence of int, optional
        Stratum label per index; defaults to ``range(n)``, i.e. one stratum per
        matched block, which is the design actually used.

    Returns
    -------
    dict
        ``chi2``, ``p_value``, ``score``, ``variance``, ``n_strata``,
        ``n_informative_strata``, ``hazard_ratio_approx``.
    """
    ta = np.asarray(ta, dtype=float)
    tb = np.asarray(tb, dtype=float)
    ea = np.asarray(ea, dtype=int)
    eb = np.asarray(eb, dtype=int)
    if strata is None:
        strata = np.arange(ta.size)
    strata = np.asarray(strata)

    score = 0.0
    var = 0.0
    o_tot = 0.0
    e_tot = 0.0
    n_inf = 0
    for s in np.unique(strata):
        m = strata == s
        t1, e1 = ta[m], ea[m]
        t2, e2 = tb[m], eb[m]
        ev = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
        contributed = False
        for t in ev:
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
            score += d1 - d * n1 / n
            o_tot += d1
            e_tot += d * n1 / n
            var += d * (n - d) * n1 * n2 / (n * n * (n - 1))
            contributed = True
        if contributed:
            n_inf += 1
    if var <= 0:
        return {"test": "stratified_logrank", "chi2": 0.0, "p_value": 1.0,
                "score": float(score), "variance": 0.0,
                "n_strata": int(np.unique(strata).size),
                "n_informative_strata": n_inf, "hazard_ratio_approx": 1.0}
    chi2 = score * score / var
    hr = float(math.exp(score / var)) if var > 0 else 1.0
    return {
        "test": "stratified_logrank",
        "chi2": float(chi2),
        "p_value": float(stats.chi2.sf(chi2, df=1)),
        "score": float(score),
        "variance": float(var),
        "observed_a": float(o_tot),
        "expected_a": float(e_tot),
        "n_strata": int(np.unique(strata).size),
        "n_informative_strata": n_inf,
        "hazard_ratio_approx": hr,
    }


def paired_prob_superiority(
    ta: Sequence[float],
    ea: Sequence[int],
    tb: Sequence[float],
    eb: Sequence[int],
    n_boot: int = 10_000,
    seed: int = 20260804,
) -> Dict[str, float]:
    """Within-block probability that arm ``a`` beats arm ``b``, with a bootstrap CI.

    This is the matched-design analogue of the Vargha-Delaney statistic: instead
    of comparing all ``n**2`` cross-arm pairs it compares only the ``n`` pairs the
    design actually matched. Ties contribute ``1/2``; indeterminate (censoring-
    ambiguous) blocks are excluded and their count reported.

    Parameters
    ----------
    ta, tb : sequence of float
        Times per block.
    ea, eb : sequence of int
        Event indicators.
    n_boot : int, optional
        Bootstrap replicates over blocks, default 10000.
    seed : int, optional
        RNG seed.

    Returns
    -------
    dict
        ``p_superior``, ``ci_lo``, ``ci_hi``, ``n_comparable``, ``n_excluded``.
    """
    ta = np.asarray(ta, dtype=float)
    tb = np.asarray(tb, dtype=float)
    ea = np.asarray(ea, dtype=int)
    eb = np.asarray(eb, dtype=int)
    scores = []
    for i in range(ta.size):
        if ea[i] and eb[i]:
            scores.append(1.0 if ta[i] < tb[i] else (0.0 if tb[i] < ta[i] else 0.5))
        elif ea[i] and not eb[i]:
            scores.append(1.0 if ta[i] <= tb[i] else math.nan)
        elif eb[i] and not ea[i]:
            scores.append(0.0 if tb[i] <= ta[i] else math.nan)
        else:
            scores.append(math.nan)
    arr = np.asarray(scores, dtype=float)
    ok = arr[~np.isnan(arr)]
    if ok.size == 0:
        return {"p_superior": 0.5, "ci_lo": 0.0, "ci_hi": 1.0,
                "n_comparable": 0, "n_excluded": int(arr.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ok.size, size=(n_boot, ok.size))
    boots = ok[idx].mean(axis=1)
    return {
        "p_superior": float(ok.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n_comparable": int(ok.size),
        "n_excluded": int(arr.size - ok.size),
    }


def sanitize_nonfinite(node):
    """Replace non-finite floats so the document is strict RFC 8259 JSON.

    Bare ``Infinity`` / ``NaN`` tokens are a Python-specific extension and are
    rejected by conforming JSON parsers. Degenerate-but-meaningful estimates --
    e.g. a conditional odds ratio ``n_10 / n_01`` with an empty discordant cell,
    or a win ratio with zero losses -- are therefore emitted as ``null`` with a
    companion ``"<field>_unbounded"`` flag recording the direction, so no
    information is lost and downstream consumers cannot silently misread the
    value as a finite number.

    Parameters
    ----------
    node : Any
        Arbitrarily nested JSON-like structure.

    Returns
    -------
    Any
        The same structure with every non-finite float replaced.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
                if math.isnan(value):
                    out[f"{key}_undefined"] = True
                else:
                    out[f"{key}_unbounded"] = "+inf" if value > 0 else "-inf"
            else:
                out[key] = sanitize_nonfinite(value)
        return out
    if isinstance(node, list):
        return [sanitize_nonfinite(v) for v in node]
    if isinstance(node, float) and not math.isfinite(node):
        return None
    return node
