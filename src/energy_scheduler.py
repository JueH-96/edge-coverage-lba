"""Adaptive multi-dimensional energy scheduling and dynamic context-depth scaling.

Step 1 of this study established that the 3D guidance engine (D0 edge coverage,
D1 calling context, D2 value-range data flow, D3 surprisal state machine) induces
a strict *refinement* of the AFL edge partition. Refinement is a statement about
what the abstraction can *distinguish*; it says nothing about what the fuzzer
*does* with the extra distinctions. The Step 1 context-depth sweep made the gap
concrete: at depth ``N = 16`` the D1 partition is finest, yet time-to-bug was
*worse* than at ``N = 4``, because every additional distinction admits another
corpus entry and a round-robin scheduler splits the mutation budget evenly across
a corpus that has grown without becoming more useful. That is corpus dilution,
and it is a property of the *scheduler*, not of the abstraction.

This module supplies the two mechanisms that decouple the two concerns.

Energy scheduling
-----------------
An AFL-family fuzzer does not spend one mutation per queue entry; it selects an
entry and grants it a burst of ``p_s`` mutations (its *energy*) before moving on.
The power schedule ``p_s`` is where a fuzzer expresses its priorities. Four are
implemented here so that the comparison is against real published policies rather
than a strawman:

* :class:`UniformRoundRobin` - constant energy, cyclic selection. The Step 1
  policy and the control condition.
* :class:`AFLFastExponential` - the ``FAST`` schedule of Bohme et al. (CCS'16),
  ``p_s = E0 * 2^{s(i)} / f(i)`` clipped to ``[1, M]``, where ``s(i)`` counts how
  often entry ``i`` was chosen and ``f(i)`` counts how often the *path* it
  exercises has been hit. This is the state-of-the-art single-dimensional power
  schedule and the strongest baseline.
* :class:`Static3D` - energy proportional to the fused 3D novelty score recorded
  at admission time, with fixed per-dimension weights. Isolates "use the extra
  dimensions" from "adapt to them".
* :class:`Adaptive3D` - the contribution. Per-dimension novelty is normalized by
  an online scale estimate so a high-cardinality dimension cannot swamp a
  low-cardinality one, the per-dimension weights are re-estimated online from
  realized discovery yield (a bandit-style credit assignment), and D3 surprisal
  enters multiplicatively.

Dynamic context depth
---------------------
:class:`DynamicContextManager` starts the D1 window shallow (``N = 4``) and
escalates along the ladder ``{2, 4, 8, 16}`` only when the shallow abstraction has
*saturated* - when the rate of newly discovered contexts has decayed to a small
fraction of its own peak. This is the direct fix for the Step 1 finding: the cost
of a deep window is paid only once the cheap window has stopped yielding.

Notes
-----
Every schedule exposes the same three-call interface (``admit`` / ``select`` /
``report``) and differs *only* in how ``energy`` is computed and how ``select``
orders the corpus, so an experiment that swaps schedules changes exactly one
thing. All schedules are seeded from an explicit :class:`random.Random` so that a
trial is bit-reproducible.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ENERGY_MIN",
    "ENERGY_MAX",
    "ENERGY_BASE",
    "DEPTH_LADDER",
    "CorpusEntry",
    "BaseSchedule",
    "UniformRoundRobin",
    "AFLFastExponential",
    "Static3D",
    "Adaptive3D",
    "DynamicContextManager",
    "SCHEDULES",
    "make_schedule",
    "gini",
]

# --------------------------------------------------------------------------- #
# Energy bounds
# --------------------------------------------------------------------------- #
#: Minimum energy granted to any selected entry. A floor of 1 is what makes the
#: fairness argument work: no entry can be selected and then given zero work, so
#: selection always makes progress and the priority of the selected entry always
#: strictly decreases.
ENERGY_MIN = 1
#: Maximum energy for a single burst. AFL clips its power schedules for the same
#: reason: an unclipped exponential term lets one queue entry monopolise the
#: budget after a handful of selections, which is a failure mode, not a feature.
ENERGY_MAX = 64
#: Energy granted by the uniform control, and the multiplicative base for every
#: other schedule. Chosen so the control cycles the corpus at a realistic rate.
ENERGY_BASE = 8

#: Admissible calling-context window depths, ascending. Powers of two spanning the
#: range swept in Step 1; ``N = 4`` was the Step 1 optimum and ``N = 16`` the
#: dilution-dominated end.
DEPTH_LADDER: Tuple[int, ...] = (2, 4, 8, 16)


def gini(values: Sequence[float]) -> float:
    """Gini coefficient of a non-negative distribution.

    Used to quantify *how concentrated* a schedule's energy allocation is: 0.0
    means every corpus entry received identical energy (round-robin), values near
    1.0 mean a few entries absorbed nearly the whole budget.

    Parameters
    ----------
    values : sequence of float
        Non-negative quantities (here, total energy per corpus entry).

    Returns
    -------
    float
        Gini coefficient in ``[0, 1]``; 0.0 for an empty or all-zero input.

    Raises
    ------
    ValueError
        If any value is negative.
    """
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("gini requires non-negative values")
    total = math.fsum(xs)
    if total <= 0.0:
        return 0.0
    # G = (2 * sum_i i*x_i) / (n * sum_i x_i) - (n + 1) / n, with i 1-based.
    weighted = math.fsum((i + 1) * x for i, x in enumerate(xs))
    g = (2.0 * weighted) / (n * total) - (n + 1.0) / n
    # Clamp: floating-point error can push a degenerate all-equal case to -1e-17.
    return min(1.0, max(0.0, g))


# --------------------------------------------------------------------------- #
# Corpus entry
# --------------------------------------------------------------------------- #
@dataclass
class CorpusEntry:
    """One queue entry, with the guidance evidence that admitted it.

    Attributes
    ----------
    index : int
        Position in the corpus list.
    data : bytes
        The input itself.
    novelty : dict
        ``dimension -> number of newly discovered exact elements`` on the
        execution that admitted this entry. This is the raw, unnormalized
        per-dimension evidence; every 3D schedule is a different function of it.
    surprisal_bits : float
        D3 surprisal ``S(s) = -log2 P(tau_i)`` accumulated over the admitting
        execution (0.0 when D3 is disabled).
    fused_score : float
        The engine's own fused novelty score for the admitting execution.
    path_key : int
        Hash of the coverage signature, used by :class:`AFLFastExponential` to
        count how often this *path* has been fuzzed (AFLFast's ``f(i)``).
    depth_at_admission : int
        D1 context depth in force when this entry was admitted; recorded so the
        analysis can attribute corpus growth to depth escalations.
    admitted_exec : int
        Execution index at which the entry entered the queue.
    n_fuzz : int
        Number of times this entry has been *selected*, i.e. AFLFast's ``s(i)``.
    energy_spent : int
        Total mutations spent on this entry across all its bursts.
    n_children : int
        Number of descendants of this entry that were themselves interesting.
        An entry with ``n_children == 0`` after being fuzzed is *sterile*: it
        occupied budget and returned nothing. The sterile fraction is this
        study's operational measure of corpus dilution.
    """

    index: int
    data: bytes
    novelty: Dict[str, int] = field(default_factory=dict)
    surprisal_bits: float = 0.0
    fused_score: float = 0.0
    path_key: int = 0
    depth_at_admission: int = 0
    admitted_exec: int = 0
    n_fuzz: int = 0
    energy_spent: int = 0
    n_children: int = 0


# --------------------------------------------------------------------------- #
# Base schedule
# --------------------------------------------------------------------------- #
class BaseSchedule:
    """Common queue bookkeeping; subclasses define energy and selection order.

    The contract is deliberately narrow. A fuzzing loop calls

    1. :meth:`admit` once per new corpus entry,
    2. :meth:`select` to obtain ``(index, energy)`` for the next burst,
    3. :meth:`report` once at the end of the burst with what the burst produced.

    Anything a subclass wants to adapt on must be derivable from those three
    calls, which is what keeps the schedules comparable.

    Parameters
    ----------
    rng : random.Random
        Seeded RNG; used only by schedules with a stochastic tie-break.
    dims : sequence of str, optional
        Guidance dimensions in play. Determines which novelty channels the 3D
        schedules read. Default ``("D0",)``.
    """

    #: Human-readable schedule name, used as the experimental condition label.
    NAME = "base"

    def __init__(self, rng: random.Random, dims: Sequence[str] = ("D0",)) -> None:
        self.rng = rng
        self.dims = tuple(dims)
        self.entries: List[CorpusEntry] = []
        #: path_key -> number of bursts spent on entries exercising that path
        self.path_freq: Dict[int, int] = {}
        self.n_selections = 0
        self.total_energy = 0
        #: (exec_index, entry_index, energy) for the energy-allocation figure
        self.energy_log: List[Tuple[int, int, int]] = []

    # -- queue management --------------------------------------------------- #
    def admit(self, entry: CorpusEntry) -> int:
        """Add ``entry`` to the queue and return its index."""
        entry.index = len(self.entries)
        self.entries.append(entry)
        self.path_freq.setdefault(entry.path_key, 0)
        self._on_admit(entry)
        return entry.index

    def _on_admit(self, entry: CorpusEntry) -> None:
        """Subclass hook fired after an entry joins the queue."""

    # -- selection ---------------------------------------------------------- #
    def energy(self, entry: CorpusEntry) -> int:
        """Mutations to grant ``entry`` for its next burst.

        Returns
        -------
        int
            Integer in ``[ENERGY_MIN, ENERGY_MAX]``.
        """
        raise NotImplementedError

    def _pick(self) -> int:
        """Index of the next entry to fuzz."""
        raise NotImplementedError

    def select(self, exec_index: int = 0) -> Tuple[int, int]:
        """Choose the next entry and its energy budget.

        Parameters
        ----------
        exec_index : int, optional
            Current execution counter, recorded in the energy log.

        Returns
        -------
        tuple of (int, int)
            ``(corpus_index, energy)``.

        Raises
        ------
        RuntimeError
            If the corpus is empty.
        """
        if not self.entries:
            raise RuntimeError("cannot select from an empty corpus")
        idx = self._pick()
        e = int(self.energy(self.entries[idx]))
        # The clip is enforced here, once, so no subclass can violate the bound
        # that the invariant tests and the fairness argument both rely on.
        e = max(ENERGY_MIN, min(ENERGY_MAX, e))
        ent = self.entries[idx]
        ent.n_fuzz += 1
        ent.energy_spent += e
        self.path_freq[ent.path_key] = self.path_freq.get(ent.path_key, 0) + 1
        self.n_selections += 1
        self.total_energy += e
        self.energy_log.append((exec_index, idx, e))
        return idx, e

    # -- feedback ----------------------------------------------------------- #
    def report(
        self,
        idx: int,
        n_new_children: int,
        dim_yield: Optional[Dict[str, int]] = None,
        energy_used: int = 0,
    ) -> None:
        """Record what a finished burst produced.

        Parameters
        ----------
        idx : int
            Entry that was fuzzed.
        n_new_children : int
            Interesting descendants the burst produced.
        dim_yield : dict, optional
            ``dimension -> new exact elements`` summed over the burst; consumed
            by :class:`Adaptive3D` for weight credit assignment.
        energy_used : int, optional
            Mutations actually executed (may be less than granted if the trial
            hit its budget mid-burst).
        """
        self.entries[idx].n_children += n_new_children
        self._on_report(idx, n_new_children, dim_yield or {}, energy_used)

    def _on_report(
        self, idx: int, n_new_children: int, dim_yield: Dict[str, int], energy_used: int
    ) -> None:
        """Subclass hook fired after a burst completes."""

    # -- diagnostics -------------------------------------------------------- #
    def sterile_fraction(self) -> float:
        """Fraction of *fuzzed* entries that produced no interesting descendant.

        This is the study's operational corpus-dilution metric. Entries that were
        never selected are excluded: they cost queue memory but no mutation
        budget, so counting them would conflate two different costs.

        Returns
        -------
        float
            Value in ``[0, 1]``; 0.0 when no entry has been fuzzed yet.
        """
        fuzzed = [e for e in self.entries if e.n_fuzz > 0]
        if not fuzzed:
            return 0.0
        return sum(1 for e in fuzzed if e.n_children == 0) / len(fuzzed)

    def energy_gini(self) -> float:
        """Gini coefficient of cumulative energy across all corpus entries."""
        return gini([e.energy_spent for e in self.entries])

    def stats(self) -> Dict[str, Any]:
        """Schedule-level summary for the results record."""
        spent = [e.energy_spent for e in self.entries]
        fuzzed = [e for e in self.entries if e.n_fuzz > 0]
        return {
            "schedule": self.NAME,
            "corpus_size": len(self.entries),
            "n_selections": self.n_selections,
            "total_energy": self.total_energy,
            "mean_energy_per_selection": (
                self.total_energy / self.n_selections if self.n_selections else 0.0
            ),
            "energy_gini": self.energy_gini(),
            "max_entry_energy": max(spent) if spent else 0,
            "n_entries_fuzzed": len(fuzzed),
            "n_entries_never_fuzzed": len(self.entries) - len(fuzzed),
            "sterile_fraction": self.sterile_fraction(),
        }


# --------------------------------------------------------------------------- #
# Baseline 1 - uniform round robin (the Step 1 control)
# --------------------------------------------------------------------------- #
class UniformRoundRobin(BaseSchedule):
    """Constant energy, cyclic selection.

    The policy Step 1 ran under. Every entry receives exactly
    :data:`ENERGY_BASE` mutations per queue cycle regardless of why it was
    admitted, so a corpus that doubles in size halves the attention paid to every
    member of it. Included as the control against which dilution mitigation is
    measured.
    """

    NAME = "uniform_rr"

    def __init__(self, rng: random.Random, dims: Sequence[str] = ("D0",)) -> None:
        super().__init__(rng, dims)
        self._cursor = 0

    def energy(self, entry: CorpusEntry) -> int:
        """Constant :data:`ENERGY_BASE`, independent of ``entry``."""
        return ENERGY_BASE

    def _pick(self) -> int:
        idx = self._cursor % len(self.entries)
        self._cursor += 1
        return idx


# --------------------------------------------------------------------------- #
# Baseline 2 - AFLFast FAST power schedule
# --------------------------------------------------------------------------- #
class AFLFastExponential(BaseSchedule):
    """The ``FAST`` power schedule of Bohme et al., "Coverage-based Greybox
    Fuzzing as Markov Chain" (CCS'16), as shipped in AFL++.

    Energy is

    .. math::

        p_s = \\mathrm{clip}\\!\\left(E_0 \\cdot \\frac{2^{\\min(s(i),\\,S_{max})}}
                                          {f(i)},\\; 1,\\; M\\right)

    where ``s(i)`` is the number of times entry ``i`` has been selected and
    ``f(i)`` is the number of bursts spent on entries exercising the same
    execution path. The exponential numerator front-loads effort onto an entry
    that has just been discovered; the ``f(i)`` denominator is the decay that
    starves paths the fuzzer already knows well. Selection order stays cyclic,
    exactly as in AFLFast - the schedule changes energy, not order.

    Parameters
    ----------
    rng : random.Random
        Seeded RNG (unused; kept for interface uniformity).
    dims : sequence of str, optional
        Dimensions in play. FAST is single-dimensional and ignores them.
    exp_cap : int, optional
        Cap ``S_max`` on the exponent, default 12. Without it ``2^{s(i)}``
        overflows any sane energy bound after a few dozen selections; AFL++
        applies the same guard.
    """

    NAME = "afl_fast"

    def __init__(
        self, rng: random.Random, dims: Sequence[str] = ("D0",), exp_cap: int = 12
    ) -> None:
        super().__init__(rng, dims)
        self._cursor = 0
        self.exp_cap = int(exp_cap)

    def energy(self, entry: CorpusEntry) -> int:
        """FAST energy for ``entry`` (pre-clip; the clip happens in ``select``)."""
        s_i = min(entry.n_fuzz, self.exp_cap)
        f_i = max(1, self.path_freq.get(entry.path_key, 0))
        return int(ENERGY_BASE * (2.0**s_i) / f_i)

    def _pick(self) -> int:
        idx = self._cursor % len(self.entries)
        self._cursor += 1
        return idx


# --------------------------------------------------------------------------- #
# Baseline 3 - static 3D schedule
# --------------------------------------------------------------------------- #
class Static3D(BaseSchedule):
    """Energy proportional to the fused 3D novelty score, with fixed weights.

    The ablation that separates "consult the extra dimensions" from "adapt to
    them". Energy is

    .. math::

        p_s = \\mathrm{clip}\\!\\left(
              E_0 \\cdot \\frac{1 + \\sum_d w_d \\log(1 + \\nu_d(s))}
                              {1 + n_{\\mathrm{fuzz}}(s)}, 1, M\\right)

    with ``w_d`` fixed at 1.0. The ``log1p`` is the same variance-stabilizing
    transform the guidance engine uses when fusing dimensions, and the
    ``1 + n_fuzz`` denominator is the least-fuzzed-first decay that AFL applies
    to its own queue. Note what is *missing* relative to :class:`Adaptive3D`:
    the raw counts ``\\nu_d`` enter unnormalized, so a dimension whose element
    space is orders of magnitude larger (D1 at depth 16) contributes an
    order-of-magnitude larger term for reasons that have nothing to do with how
    useful it is.
    """

    NAME = "static_3d"

    def __init__(
        self,
        rng: random.Random,
        dims: Sequence[str] = ("D0", "D1", "D2", "D3"),
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(rng, dims)
        self._cursor = 0
        self.weights = {d: 1.0 for d in self.dims}
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})

    def energy(self, entry: CorpusEntry) -> int:
        """Fixed-weight fused-novelty energy."""
        s = 0.0
        for d in self.dims:
            s += self.weights.get(d, 1.0) * math.log1p(float(entry.novelty.get(d, 0)))
        return int(ENERGY_BASE * (1.0 + s) / (1.0 + entry.n_fuzz))

    def _pick(self) -> int:
        idx = self._cursor % len(self.entries)
        self._cursor += 1
        return idx


# --------------------------------------------------------------------------- #
# The contribution - adaptive 3D energy schedule
# --------------------------------------------------------------------------- #
class Adaptive3D(BaseSchedule):
    """Adaptive multi-dimensional power schedule with online weight estimation.

    Three things are wrong with :class:`Static3D`, and this class fixes each one.

    **1. Cross-dimensional scale.** ``\\nu_{D1}`` at depth 16 routinely reports
    hundreds of new elements where ``\\nu_{D0}`` reports one, purely because the
    context space is larger. A sum of raw ``log1p`` terms is therefore a sum of
    incommensurable quantities. Each dimension is instead passed through a
    saturating transform against its own online scale estimate,

    .. math:: \\tilde{\\nu}_d(s) = \\frac{\\nu_d(s)}{\\nu_d(s) + m_d}

    where ``m_d`` is an exponentially weighted mean of the non-zero novelty
    counts seen in dimension ``d``. The result lies in ``[0, 1)`` for every
    dimension, is 0.5 at the dimension's own typical magnitude, and is invariant
    to rescaling ``\\nu_d`` - which is exactly the invariance the raw sum lacks.

    **2. Fixed weights.** Which dimension is informative is a property of the
    *target*, and is not known in advance. Weights are re-estimated online from
    realized yield: when a burst on entry ``s`` produces new coverage, the credit
    is attributed to the dimensions that admitted ``s``, in proportion to their
    normalized novelty. Weight ``w_d`` tracks an EWMA of yield-per-unit-energy
    for dimension ``d``, renormalized so ``\\sum_d w_d = |D|``; a dimension that
    keeps admitting sterile seeds is demoted, one that pays off is promoted. This
    is Thompson-free, deterministic bandit credit assignment - chosen over a
    stochastic bandit precisely because it keeps a trial bit-reproducible.

    **3. Surprisal is thrown away.** D3 already computes
    ``S(s) = -log2 P(tau_i)`` under a Jeffreys-smoothed transition model, and
    Static3D collapses it into an element count. Here it enters multiplicatively
    through ``1 + lambda * S(s) / (1 + S(s))``, a bounded transform so that a
    single astronomically surprising transition cannot buy unbounded energy.

    The full schedule is

    .. math::

        p_s = \\mathrm{clip}\\!\\left(
            E_0 \\cdot
            \\frac{\\left(1 + \\sum_d w_d\\, \\tilde{\\nu}_d(s)\\right)
                   \\left(1 + \\lambda \\frac{S(s)}{1 + S(s)}\\right)}
                 {(1 + n_{\\mathrm{fuzz}}(s))^{\\gamma}},\\; 1,\\; M \\right).

    Selection is priority-first rather than cyclic: the entry with the largest
    current energy is fuzzed next, via a lazily-revalidated max-heap. Because
    energy is strictly decreasing in ``n_fuzz`` (``\\gamma > 0``) and bounded
    below by :data:`ENERGY_MIN`, no entry can be starved indefinitely - the
    priority of any repeatedly-selected entry falls below that of any waiting
    entry after finitely many selections. :meth:`starvation_bound` states the
    explicit bound that ``test_step2_scheduler.py`` checks empirically.

    Parameters
    ----------
    rng : random.Random
        Seeded RNG for deterministic tie-breaking.
    dims : sequence of str, optional
        Dimensions to fuse. Default all four.
    gamma : float, optional
        Exponent of the least-fuzzed-first decay, default 1.0.
    lam : float, optional
        Surprisal gain ``lambda``, default 0.5.
    ewma_alpha : float, optional
        Smoothing factor for both the scale estimates ``m_d`` and the weights,
        default 0.1.
    weight_floor : float, optional
        Lower bound on any weight, default 0.1. A dimension is never demoted to
        exactly zero: an unlucky early run would otherwise permanently blind the
        scheduler to a dimension that becomes informative later.
    """

    NAME = "adaptive_3d"

    def __init__(
        self,
        rng: random.Random,
        dims: Sequence[str] = ("D0", "D1", "D2", "D3"),
        gamma: float = 1.0,
        lam: float = 0.5,
        ewma_alpha: float = 0.1,
        weight_floor: float = 0.1,
    ) -> None:
        super().__init__(rng, dims)
        if gamma <= 0.0:
            raise ValueError("gamma must be positive for the fairness bound to hold")
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must lie in (0, 1]")
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.alpha = float(ewma_alpha)
        self.weight_floor = float(weight_floor)
        #: per-dimension online scale estimate m_d, initialised to 1.0
        self.scale: Dict[str, float] = {d: 1.0 for d in self.dims}
        #: per-dimension weights, initialised uniform and renormalised to sum |D|
        self.weights: Dict[str, float] = {d: 1.0 for d in self.dims}
        #: per-dimension EWMA of realized yield per unit energy
        self.yield_ewma: Dict[str, float] = {d: 0.0 for d in self.dims}
        self._heap: List[Tuple[float, int, int]] = []
        self._tick = 0
        #: trajectory of the weight vector, sampled on every weight update
        self.weight_trace: List[Tuple[int, Dict[str, float]]] = []

    # -- normalization ------------------------------------------------------ #
    def normalized_novelty(self, entry: CorpusEntry, dim: str) -> float:
        """Saturating, scale-free novelty of ``entry`` in dimension ``dim``.

        Returns
        -------
        float
            Value in ``[0, 1)``: ``nu / (nu + m_d)``.
        """
        nu = float(entry.novelty.get(dim, 0))
        if nu <= 0.0:
            return 0.0
        m = max(self.scale.get(dim, 1.0), 1e-9)
        return nu / (nu + m)

    def _update_scale(self, entry: CorpusEntry) -> None:
        """Fold a newly admitted entry's novelty into the scale estimates.

        Only non-zero observations update ``m_d``: the zeros carry no information
        about the *magnitude* of a discovery in that dimension, and including
        them would drive every scale toward zero and saturate the transform.
        """
        for d in self.dims:
            nu = float(entry.novelty.get(d, 0))
            if nu > 0.0:
                self.scale[d] = (1.0 - self.alpha) * self.scale[d] + self.alpha * nu

    # -- energy ------------------------------------------------------------- #
    def energy(self, entry: CorpusEntry) -> int:
        """Adaptive fused energy for ``entry`` (pre-clip)."""
        return int(self.energy_real(entry))

    def energy_real(self, entry: CorpusEntry) -> float:
        """Unrounded, unclipped energy - the quantity the invariants are about.

        Returns
        -------
        float
            Strictly positive real energy before integer truncation and clipping.
        """
        fused = 0.0
        for d in self.dims:
            fused += self.weights[d] * self.normalized_novelty(entry, d)
        s = max(0.0, float(entry.surprisal_bits))
        surprise = 1.0 + self.lam * (s / (1.0 + s))
        decay = (1.0 + entry.n_fuzz) ** self.gamma
        return ENERGY_BASE * (1.0 + fused) * surprise / decay

    def priority(self, entry: CorpusEntry) -> float:
        """Selection priority: the clipped energy the entry would receive."""
        return max(float(ENERGY_MIN), min(float(ENERGY_MAX), self.energy_real(entry)))

    def starvation_bound(self) -> int:
        """Selections after which every current entry must have been chosen once.

        A never-selected entry has priority at least :data:`ENERGY_MIN`. An entry
        selected ``k`` times has priority at most
        ``ENERGY_MAX_UNCLIPPED / (1 + k)^gamma`` where the unclipped numerator is
        bounded by ``E0 * (1 + sum_d w_d) * (1 + lambda)``. Setting that below
        ``ENERGY_MIN`` and solving for ``k`` bounds how many times any single
        entry can be selected before it must yield to a waiting one; multiplying
        by the corpus size bounds the whole sweep.

        Returns
        -------
        int
            Upper bound on selections until full coverage of the current corpus.
        """
        w_sum = sum(self.weights.values())
        num = ENERGY_BASE * (1.0 + w_sum) * (1.0 + self.lam)
        num = min(num, float(ENERGY_MAX))
        # (1+k)^gamma > num / ENERGY_MIN  =>  k > (num/E_MIN)^(1/gamma) - 1
        k = math.ceil((num / ENERGY_MIN) ** (1.0 / self.gamma))
        return int(k * max(1, len(self.entries)))

    # -- queue hooks -------------------------------------------------------- #
    def _on_admit(self, entry: CorpusEntry) -> None:
        self._update_scale(entry)
        self._tick += 1
        heapq.heappush(self._heap, (-self.priority(entry), self._tick, entry.index))

    def _pick(self) -> int:
        """Pop the highest-priority entry, revalidating stale heap keys lazily.

        Priorities change whenever weights or scales move, so a heap key can be
        stale. Rather than rebuild the heap on every update - which would make
        selection ``O(n log n)`` - a popped key is compared against the entry's
        current priority and re-pushed if it has drifted. The loop terminates
        because each re-push strictly lowers the stored priority toward the
        current value, and priorities are bounded below by :data:`ENERGY_MIN`.
        """
        while True:
            negp, _, idx = heapq.heappop(self._heap)
            cur = self.priority(self.entries[idx])
            if -negp <= cur + 1e-9:
                # Key is current (or conservative): accept it. Re-push with the
                # post-selection priority once select() has incremented n_fuzz -
                # done here by predicting the increment.
                ent = self.entries[idx]
                ent.n_fuzz += 1
                nxt = self.priority(ent)
                ent.n_fuzz -= 1
                self._tick += 1
                heapq.heappush(self._heap, (-nxt, self._tick, idx))
                return idx
            self._tick += 1
            heapq.heappush(self._heap, (-cur, self._tick, idx))

    def _on_report(
        self, idx: int, n_new_children: int, dim_yield: Dict[str, int], energy_used: int
    ) -> None:
        """Attribute realized yield to the dimensions that admitted the entry.

        The credit signal is ``children per unit energy``, not raw children:
        without dividing by the energy actually spent, a schedule that grants
        more energy to a dimension would manufacture evidence that the dimension
        deserves more energy - a self-fulfilling loop. Dividing by energy makes
        the statistic a *rate*, which is the quantity the schedule should be
        comparing across dimensions.
        """
        ent = self.entries[idx]
        e = max(1, int(energy_used))
        rate = float(n_new_children) / e
        share_tot = 0.0
        shares: Dict[str, float] = {}
        for d in self.dims:
            v = self.normalized_novelty(ent, d)
            shares[d] = v
            share_tot += v
        if share_tot <= 0.0:
            return
        for d in self.dims:
            credit = rate * (shares[d] / share_tot)
            self.yield_ewma[d] = (1.0 - self.alpha) * self.yield_ewma[d] + self.alpha * credit
        self._renormalize_weights()

    def _renormalize_weights(self) -> None:
        """Map the yield EWMAs onto weights summing to ``|D|`` with a floor."""
        n = len(self.dims)
        tot = sum(self.yield_ewma[d] for d in self.dims)
        if tot <= 0.0:
            new = {d: 1.0 for d in self.dims}
        else:
            raw = {d: self.weight_floor + n * self.yield_ewma[d] / tot for d in self.dims}
            s = sum(raw.values())
            new = {d: n * raw[d] / s for d in self.dims}
        self.weights = new
        self.weight_trace.append((self.n_selections, dict(new)))

    def stats(self) -> Dict[str, Any]:
        """Schedule summary plus the learned weights and scales."""
        base = super().stats()
        base["final_weights"] = {d: float(self.weights[d]) for d in self.dims}
        base["final_scales"] = {d: float(self.scale[d]) for d in self.dims}
        base["yield_ewma"] = {d: float(self.yield_ewma[d]) for d in self.dims}
        base["gamma"] = self.gamma
        base["lambda"] = self.lam
        return base


# --------------------------------------------------------------------------- #
# Dynamic context-depth manager
# --------------------------------------------------------------------------- #
class DynamicContextManager:
    """Escalate the D1 calling-context window only once it has saturated.

    Step 1 swept the window depth ``N`` as a fixed hyper-parameter and found
    ``N = 4`` significantly faster than the control while ``N = 16`` was not:
    the deeper window distinguishes more, admits more, and dilutes the mutation
    budget across a corpus whose extra members carry no extra reachability. But
    a fixed shallow window is only right *until the shallow abstraction runs
    out*; past that point the fuzzer has nothing left to climb.

    This manager starts at ``N = 4`` and escalates one rung when the shallow
    window is demonstrably exhausted. Saturation is measured on the *discovery
    rate* rather than on a raw count: over a sliding window of ``window``
    executions, the manager records how many previously unseen contexts appeared.
    Let ``r_t`` be that count and ``r_peak`` its running maximum. The depth is
    escalated when

    * ``r_t <= saturation_ratio * r_peak`` for ``patience`` consecutive windows,
    * at least ``min_execs_at_depth`` executions have run at the current depth,
      which stops a slow start being mistaken for saturation, and
    * the ladder has a next rung.

    Escalation rebuilds the D1 tracker at the new depth. That is a real cost -
    D1 coverage is re-learned from scratch - and it is *sound* rather than merely
    convenient: the depth-``N`` context partition refines the depth-``N'`` one for
    ``N > N'``, so re-learning re-derives every distinction the coarser window had
    plus new ones, and never silently loses a distinction that had been claimed.
    The re-learning cost is precisely why escalation is gated on saturation
    instead of being applied from the start.

    Parameters
    ----------
    ladder : sequence of int, optional
        Admissible depths, ascending. Default :data:`DEPTH_LADDER`.
    start_depth : int, optional
        Initial depth, default 4. Must be in ``ladder``.
    window : int, optional
        Executions per saturation window, default 500.
    saturation_ratio : float, optional
        Fraction of peak discovery rate below which a window counts as
        saturated, default 0.05.
    patience : int, optional
        Consecutive saturated windows required to escalate, default 2.
    min_execs_at_depth : int, optional
        Executions that must elapse at a depth before it may be left,
        default 1000.

    Raises
    ------
    ValueError
        If ``ladder`` is not strictly ascending, ``start_depth`` is not in it, or
        ``saturation_ratio`` is outside ``(0, 1)``.
    """

    def __init__(
        self,
        ladder: Sequence[int] = DEPTH_LADDER,
        start_depth: int = 4,
        window: int = 500,
        saturation_ratio: float = 0.05,
        patience: int = 2,
        min_execs_at_depth: int = 1000,
    ) -> None:
        lad = tuple(int(x) for x in ladder)
        if len(lad) == 0 or any(b <= a for a, b in zip(lad, lad[1:])):
            raise ValueError("ladder must be non-empty and strictly ascending")
        if start_depth not in lad:
            raise ValueError(f"start_depth {start_depth} not in ladder {lad}")
        if not 0.0 < saturation_ratio < 1.0:
            raise ValueError("saturation_ratio must lie in (0, 1)")
        self.ladder = lad
        self.window = int(window)
        self.saturation_ratio = float(saturation_ratio)
        self.patience = int(patience)
        self.min_execs_at_depth = int(min_execs_at_depth)

        self._level = lad.index(int(start_depth))
        self.n_execs = 0
        self._execs_at_depth = 0
        self._window_new = 0
        self._window_execs = 0
        self._peak_rate = 0.0
        self._saturated_windows = 0
        #: (exec_index, depth) at every change, seeded with the starting depth
        self.depth_trace: List[Tuple[int, int]] = [(0, self.depth)]
        #: (exec_index, from_depth, to_depth, rate, peak_rate) per escalation
        self.escalations: List[Tuple[int, int, int, float, float]] = []

    @property
    def depth(self) -> int:
        """Current calling-context window ``N``."""
        return self.ladder[self._level]

    @property
    def at_max_depth(self) -> bool:
        """Whether the ladder has been exhausted."""
        return self._level >= len(self.ladder) - 1

    def observe(self, n_new_contexts: int) -> bool:
        """Record one execution and decide whether to escalate.

        Parameters
        ----------
        n_new_contexts : int
            Previously unseen D1 contexts discovered by this execution.

        Returns
        -------
        bool
            True iff the depth was escalated on this call, in which case the
            caller must rebuild the D1 tracker at :attr:`depth`.

        Raises
        ------
        ValueError
            If ``n_new_contexts`` is negative.
        """
        if n_new_contexts < 0:
            raise ValueError("n_new_contexts must be non-negative")
        self.n_execs += 1
        self._execs_at_depth += 1
        self._window_new += int(n_new_contexts)
        self._window_execs += 1
        if self._window_execs < self.window:
            return False

        rate = self._window_new / float(self._window_execs)
        self._window_new = 0
        self._window_execs = 0
        prev_peak = self._peak_rate
        self._peak_rate = max(self._peak_rate, rate)

        # A peak of zero means nothing has ever been discovered at this depth;
        # that is a degenerate case, not saturation, so it is never escalated on.
        if prev_peak > 0.0 and rate <= self.saturation_ratio * prev_peak:
            self._saturated_windows += 1
        else:
            self._saturated_windows = 0

        if (
            self._saturated_windows >= self.patience
            and self._execs_at_depth >= self.min_execs_at_depth
            and not self.at_max_depth
        ):
            old = self.depth
            self._level += 1
            self._saturated_windows = 0
            self._execs_at_depth = 0
            # The peak is per-depth: a deeper window has a larger context space
            # and its own natural discovery rate, so carrying the shallow peak
            # forward would make the deep window look saturated immediately.
            self._peak_rate = 0.0
            self.depth_trace.append((self.n_execs, self.depth))
            self.escalations.append((self.n_execs, old, self.depth, rate, prev_peak))
            return True
        return False

    def stats(self) -> Dict[str, Any]:
        """Summary of the depth trajectory for the results record."""
        return {
            "ladder": list(self.ladder),
            "final_depth": self.depth,
            "n_escalations": len(self.escalations),
            "n_execs": self.n_execs,
            "depth_trace": [list(t) for t in self.depth_trace],
            "escalations": [
                {
                    "exec": e,
                    "from_depth": a,
                    "to_depth": b,
                    "window_rate": r,
                    "peak_rate": p,
                }
                for (e, a, b, r, p) in self.escalations
            ],
        }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#: schedule name -> class
SCHEDULES: Dict[str, type] = {
    UniformRoundRobin.NAME: UniformRoundRobin,
    AFLFastExponential.NAME: AFLFastExponential,
    Static3D.NAME: Static3D,
    Adaptive3D.NAME: Adaptive3D,
}


def make_schedule(
    name: str, rng: random.Random, dims: Sequence[str] = ("D0",), **kwargs: Any
) -> BaseSchedule:
    """Instantiate a schedule by name.

    Parameters
    ----------
    name : str
        Key of :data:`SCHEDULES`.
    rng : random.Random
        Seeded RNG.
    dims : sequence of str, optional
        Enabled guidance dimensions.
    **kwargs
        Forwarded to the schedule constructor.

    Returns
    -------
    BaseSchedule

    Raises
    ------
    KeyError
        If ``name`` is not a registered schedule.
    """
    if name not in SCHEDULES:
        raise KeyError(f"unknown schedule {name!r}; known: {sorted(SCHEDULES)}")
    return SCHEDULES[name](rng, dims, **kwargs)
