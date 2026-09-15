"""Invariant and unit tests for the Step 2 energy scheduler and trigram targets.

Run with ``uv run pytest -q workflow/test_step2_scheduler.py``.

The tests are grouped by the property they defend:

* **Energy bounds** - every schedule respects ``[ENERGY_MIN, ENERGY_MAX]`` and
  the monotonicity the fairness argument depends on.
* **Selection properties** - starvation-freedom, and the specific ordering each
  baseline claims to implement.
* **Adaptivity** - normalization is scale-free, weights track realized yield,
  and the schedule is bit-reproducible from its seed.
* **Context scaling** - the depth ladder is respected, escalation happens only
  on genuine saturation, and never past the top rung.
* **Target invariants** - the trigram ladders are monotone, the witness pair is
  bigram-identical with opposite bug outcomes, and partial progress emits no
  trace point (the blind-spot property the whole benchmark rests on).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

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

from energy_scheduler import (  # noqa: E402
    DEPTH_LADDER,
    ENERGY_BASE,
    ENERGY_MAX,
    ENERGY_MIN,
    Adaptive3D,
    AFLFastExponential,
    CorpusEntry,
    DynamicContextManager,
    SCHEDULES,
    Static3D,
    UniformRoundRobin,
    gini,
    make_schedule,
)
from guidance_engine import MultiDimGuidanceEngine  # noqa: E402
from trigram_targets import (  # noqa: E402
    TG1TrigramLock,
    TG2ValueTrigram,
    TG3SurprisalMaze,
    bigram_multiset,
    get_trigram_targets,
    trigram_multiset,
)

DIMS = ("D0", "D1", "D2", "D3")


def _entry(idx: int = 0, **kw) -> CorpusEntry:
    """Build a corpus entry with sensible defaults for the tests."""
    kw.setdefault("data", bytes([idx & 0xFF]) * 8)
    kw.setdefault("novelty", {d: 1 for d in DIMS})
    kw.setdefault("path_key", idx)
    return CorpusEntry(index=idx, **kw)


def _encode(word, xor: int, mask: int, length: int = 24, fill: int = 0x11) -> bytes:
    """Encode a symbol word into an input buffer, inverting the target decode."""
    buf = bytearray(fill for _ in range(length))
    for i, s in enumerate(word):
        buf[i] = ((buf[i] & ~mask & 0xFF) | s) ^ xor
    return bytes(buf)


def _solution_input(tgt) -> bytes:
    """A known bug-triggering input for ``tgt``.

    Every target's key sequence is public (it is what the ladder demands), so a
    solution can be constructed directly rather than searched for. Tests that
    need to exercise the oracle use this instead of hoping a uniform sample
    stumbles into it - which, by design, it never will.
    """
    key = getattr(tgt, "WITNESS_POS", None) or tgt.WITNESS_OPS
    mask = getattr(tgt, "MASK", None)
    if mask is None:
        mask = tgt.OP_MASK
    fill = next(
        b
        for b in range(256)
        if not hasattr(tgt, "ADVANCE") or ((b ^ tgt.DECODE_XOR) & mask) not in tgt.ADVANCE
    )
    buf = bytearray(_encode(key, tgt.DECODE_XOR, mask, length=24, fill=fill))
    if hasattr(tgt, "MAGIC"):
        buf[tgt.VAL_OFF : tgt.VAL_OFF + 4] = tgt.MAGIC.to_bytes(4, "little")
    return bytes(buf)


# --------------------------------------------------------------------------- #
# Energy bounds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_energy_within_bounds_over_a_long_run(name: str) -> None:
    """No schedule ever grants energy outside ``[ENERGY_MIN, ENERGY_MAX]``.

    The bound is enforced centrally in ``BaseSchedule.select`` precisely so that
    no subclass can violate it, and the fairness argument for
    :class:`Adaptive3D` depends on both ends of it.
    """
    rng = random.Random(11)
    sched = make_schedule(name, rng, DIMS)
    sched.admit(_entry(0))
    for step in range(2000):
        idx, e = sched.select(step)
        assert ENERGY_MIN <= e <= ENERGY_MAX, f"{name} granted {e}"
        assert isinstance(e, int)
        sched.report(idx, 1 if step % 7 == 0 else 0, {d: step % 3 for d in DIMS}, e)
        if step % 50 == 0 and len(sched.entries) < 60:
            sched.admit(
                _entry(
                    len(sched.entries),
                    novelty={d: rng.randint(0, 500) for d in DIMS},
                    surprisal_bits=rng.random() * 40.0,
                )
            )


@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_energy_is_positive_even_for_a_zero_novelty_entry(name: str) -> None:
    """An entry with no recorded novelty still receives at least one mutation.

    A zero-energy grant would make ``select`` return without doing work, which
    would both waste a selection and break the argument that every selection
    strictly lowers the selected entry's priority.
    """
    sched = make_schedule(name, random.Random(3), DIMS)
    sched.admit(_entry(0, novelty={d: 0 for d in DIMS}, surprisal_bits=0.0))
    _, e = sched.select(0)
    assert e >= ENERGY_MIN


def test_adaptive_energy_is_monotone_decreasing_in_n_fuzz() -> None:
    """More times fuzzed => never more energy. This is the decay term."""
    sched = Adaptive3D(random.Random(0), DIMS)
    prev = float("inf")
    for k in range(40):
        e = sched.energy_real(_entry(0, n_fuzz=k))
        assert e <= prev + 1e-12, f"energy rose at n_fuzz={k}"
        prev = e


def test_adaptive_energy_is_monotone_increasing_in_novelty() -> None:
    """More novelty in any dimension => never less energy."""
    sched = Adaptive3D(random.Random(0), DIMS)
    prev = -1.0
    for nu in (0, 1, 2, 5, 10, 100, 1000):
        e = sched.energy_real(_entry(0, novelty={"D0": nu, "D1": 0, "D2": 0, "D3": 0}))
        assert e >= prev - 1e-12, f"energy fell at novelty={nu}"
        prev = e


def test_adaptive_energy_is_monotone_increasing_in_surprisal() -> None:
    """More D3 surprisal => never less energy, and the effect stays bounded."""
    sched = Adaptive3D(random.Random(0), DIMS)
    vals = [sched.energy_real(_entry(0, surprisal_bits=s)) for s in (0, 1, 5, 50, 1e6)]
    assert vals == sorted(vals)
    # The surprisal factor is bounded by (1 + lambda), so an astronomically
    # surprising transition cannot buy unbounded energy.
    assert vals[-1] <= vals[0] * (1.0 + sched.lam) + 1e-9


# --------------------------------------------------------------------------- #
# Selection properties
# --------------------------------------------------------------------------- #
def test_round_robin_visits_every_entry_once_per_cycle() -> None:
    """The control policy is exactly cyclic, with constant energy."""
    sched = UniformRoundRobin(random.Random(0), ("D0",))
    for i in range(7):
        sched.admit(_entry(i))
    picks = [sched.select(t)[0] for t in range(14)]
    assert picks == list(range(7)) * 2
    assert all(e.energy_spent == 2 * ENERGY_BASE for e in sched.entries)


def test_afl_fast_energy_matches_the_published_formula() -> None:
    """``p_s = E0 * 2^{s(i)} / f(i)``, clipped - the AFLFast FAST schedule."""
    sched = AFLFastExponential(random.Random(0), ("D0",))
    sched.admit(_entry(0, path_key=42))
    for expected_s in range(6):
        ent = sched.entries[0]
        assert ent.n_fuzz == expected_s
        f_i = max(1, sched.path_freq.get(42, 0))
        want = min(ENERGY_MAX, max(ENERGY_MIN, int(ENERGY_BASE * 2.0**expected_s / f_i)))
        _, got = sched.select(expected_s)
        assert got == want


def test_afl_fast_decays_a_frequently_hit_path() -> None:
    """Two entries on the same path share ``f(i)``, so both are starved of energy.

    This is the whole point of the FAST denominator, and it is the behaviour a
    naive per-entry counter would silently fail to reproduce.
    """
    shared = AFLFastExponential(random.Random(0), ("D0",))
    for i in range(2):
        shared.admit(_entry(i, path_key=7))  # same path
    distinct = AFLFastExponential(random.Random(0), ("D0",))
    for i in range(2):
        distinct.admit(_entry(i, path_key=i))  # different paths
    for t in range(8):
        i, e = shared.select(t)
        shared.report(i, 0, {}, e)
        i, e = distinct.select(t)
        distinct.report(i, 0, {}, e)
    assert shared.total_energy < distinct.total_energy


def test_adaptive_selection_is_starvation_free() -> None:
    """Every corpus entry is selected within the stated bound.

    Priority-first selection is only admissible if it cannot starve an entry.
    Energy is strictly decreasing in ``n_fuzz`` and bounded below by
    ``ENERGY_MIN``, so a repeatedly-chosen entry must eventually fall below any
    waiting one; :meth:`Adaptive3D.starvation_bound` states the bound and this
    test checks it empirically against an adversarially skewed corpus.
    """
    sched = Adaptive3D(random.Random(5), DIMS)
    # Entry 0 is made maximally attractive; the rest are maximally dull.
    sched.admit(_entry(0, novelty={d: 10**6 for d in DIMS}, surprisal_bits=1e3))
    for i in range(1, 25):
        sched.admit(_entry(i, novelty={d: 0 for d in DIMS}, surprisal_bits=0.0))
    bound = sched.starvation_bound()
    seen = set()
    for t in range(bound):
        idx, e = sched.select(t)
        seen.add(idx)
        sched.report(idx, 0, {}, e)
        if len(seen) == len(sched.entries):
            break
    assert seen == set(range(len(sched.entries))), (
        f"starved {sorted(set(range(len(sched.entries))) - seen)} within {bound}"
    )


def test_adaptive_prefers_the_more_novel_entry_first() -> None:
    """Priority-first selection actually orders by energy on the first pass."""
    sched = Adaptive3D(random.Random(0), DIMS)
    sched.admit(_entry(0, novelty={d: 0 for d in DIMS}))
    sched.admit(_entry(1, novelty={d: 100 for d in DIMS}, surprisal_bits=20.0))
    assert sched.select(0)[0] == 1


def test_schedules_are_bit_reproducible_from_their_seed() -> None:
    """Identical seeds give identical selection and energy sequences.

    Note the stronger property this actually establishes for
    :class:`Adaptive3D`: its selection rule consults no random state at all, so
    the sequence depends only on the admit/report history. That is deliberate -
    a stochastic bandit would have been the textbook choice for the weight
    update, and it was rejected precisely because it would make a trial
    irreproducible from its seed, which the Klees et al. protocol needs.
    """

    def run(seed: int):
        sched = Adaptive3D(random.Random(seed), DIMS)
        sched.admit(_entry(0))
        out = []
        for t in range(300):
            idx, e = sched.select(t)
            out.append((idx, e))
            sched.report(idx, t % 3 == 0, {d: t % 5 for d in DIMS}, e)
            if t % 20 == 0 and len(sched.entries) < 30:
                sched.admit(_entry(len(sched.entries), novelty={d: t for d in DIMS}))
        return out

    assert run(1234) == run(1234)
    assert run(5678) == run(5678)
    assert run(1234) == run(5678)  # no random state is consulted at all


# --------------------------------------------------------------------------- #
# Adaptivity
# --------------------------------------------------------------------------- #
def test_normalized_novelty_is_scale_free() -> None:
    """Rescaling a dimension's counts leaves its normalized novelty unchanged.

    This is the invariance the raw ``log1p`` sum in :class:`Static3D` lacks, and
    the reason a high-cardinality dimension cannot swamp a low-cardinality one.
    """
    small = Adaptive3D(random.Random(0), DIMS)
    large = Adaptive3D(random.Random(0), DIMS)
    for k in range(300):
        small.admit(_entry(k, novelty={"D1": 3, "D0": 1, "D2": 1, "D3": 1}))
        large.admit(_entry(k, novelty={"D1": 3000, "D0": 1, "D2": 1, "D3": 1}))
    a = small.normalized_novelty(_entry(0, novelty={"D1": 3}), "D1")
    b = large.normalized_novelty(_entry(0, novelty={"D1": 3000}), "D1")
    assert abs(a - b) < 1e-6
    assert 0.0 <= a < 1.0


def test_normalized_novelty_is_bounded_and_zero_at_zero() -> None:
    """The saturating transform maps into ``[0, 1)`` with ``nu = 0 -> 0``."""
    sched = Adaptive3D(random.Random(0), DIMS)
    assert sched.normalized_novelty(_entry(0, novelty={"D0": 0}), "D0") == 0.0
    for nu in (1, 10, 10**9):
        v = sched.normalized_novelty(_entry(0, novelty={"D0": nu}), "D0")
        assert 0.0 < v < 1.0


def test_weights_sum_to_the_dimension_count_and_respect_the_floor() -> None:
    """The weight vector is renormalized to ``|D|`` with every weight above the floor."""
    sched = Adaptive3D(random.Random(0), DIMS)
    for i in range(10):
        sched.admit(_entry(i))
    for t in range(200):
        idx, e = sched.select(t)
        sched.report(idx, t % 4 == 0, {d: t % 6 for d in DIMS}, e)
    assert abs(sum(sched.weights.values()) - len(DIMS)) < 1e-9
    assert all(w > 0.0 for w in sched.weights.values())


def test_weights_promote_the_dimension_that_actually_pays_off() -> None:
    """A dimension whose seeds yield children is weighted above one that does not.

    Credit is assigned in proportion to normalized novelty, so an entry admitted
    solely on D3 evidence that produces children must push D3's weight above the
    weight of a dimension whose entries are consistently sterile.
    """
    sched = Adaptive3D(random.Random(0), DIMS)
    productive = _entry(0, novelty={"D0": 0, "D1": 0, "D2": 0, "D3": 10})
    sterile = _entry(1, novelty={"D0": 0, "D1": 10, "D2": 0, "D3": 0})
    sched.admit(productive)
    sched.admit(sterile)
    for _ in range(150):
        sched.report(0, 5, {}, ENERGY_BASE)  # D3 entry pays off
        sched.report(1, 0, {}, ENERGY_BASE)  # D1 entry never does
    assert sched.weights["D3"] > sched.weights["D1"]


def test_static_3d_is_not_scale_free() -> None:
    """The documented weakness of the fixed-weight baseline is real, not rhetorical.

    Recorded as a test so the contrast drawn in the paper is a measured property
    of the implementation rather than an assertion about it.
    """
    sched = Static3D(random.Random(0), DIMS)
    lo = sched.energy(_entry(0, novelty={"D0": 1, "D1": 1, "D2": 0, "D3": 0}))
    hi = sched.energy(_entry(0, novelty={"D0": 1, "D1": 1000, "D2": 0, "D3": 0}))
    assert hi > lo


# --------------------------------------------------------------------------- #
# Dynamic context depth
# --------------------------------------------------------------------------- #
def test_depth_starts_shallow_and_stays_on_the_ladder() -> None:
    """Start at N = 4; every observed depth is a ladder rung."""
    mgr = DynamicContextManager(window=10, min_execs_at_depth=10, patience=1)
    assert mgr.depth == 4
    for i in range(4000):
        mgr.observe(50 if i < 200 else 0)
        assert mgr.depth in DEPTH_LADDER


def test_depth_is_monotone_non_decreasing_and_capped_at_the_top_rung() -> None:
    """Escalation only ever goes up, and never past the last rung."""
    mgr = DynamicContextManager(window=10, min_execs_at_depth=10, patience=1)
    seen = [mgr.depth]
    # Each depth gets a discovery burst and is then starved. This mirrors what a
    # real escalation looks like - a deeper window always finds new contexts at
    # first, because it strictly refines the shallower one - and it is required
    # to traverse the ladder: the manager treats a depth that has discovered
    # NOTHING as degenerate rather than saturated, and deliberately will not
    # escalate off it, since a window that found nothing gives no evidence that
    # a deeper one would do better.
    for _ in range(len(DEPTH_LADDER)):
        for _ in range(100):
            mgr.observe(50)
        for _ in range(400):
            mgr.observe(0)
        if mgr.depth != seen[-1]:
            seen.append(mgr.depth)
    assert seen == sorted(seen), f"depth went down: {seen}"
    assert mgr.depth == DEPTH_LADDER[-1]
    assert mgr.at_max_depth
    # The ladder has 4 rungs and the manager starts on rung 1 (N = 4), so
    # exactly two escalations are available: 4 -> 8 and 8 -> 16.
    assert len(mgr.escalations) == len(DEPTH_LADDER) - DEPTH_LADDER.index(4) - 1


def test_no_escalation_while_discovery_continues() -> None:
    """A sustained discovery rate must never be mistaken for saturation."""
    mgr = DynamicContextManager(window=50, min_execs_at_depth=50, patience=2)
    for _ in range(5000):
        assert not mgr.observe(10)
    assert mgr.depth == 4
    assert mgr.escalations == []


def test_no_escalation_before_the_minimum_dwell_time() -> None:
    """A slow start is not saturation: the dwell-time guard holds the depth."""
    mgr = DynamicContextManager(window=10, min_execs_at_depth=5000, patience=1)
    for _ in range(2000):
        mgr.observe(0)
    assert mgr.depth == 4


def test_escalation_resets_the_peak_so_a_deep_window_is_not_instantly_saturated() -> None:
    """The saturation peak is per-depth.

    Carrying a shallow window's peak forward would make the deeper window look
    saturated on its first window and cascade straight to the top of the ladder,
    which would defeat the entire mechanism.
    """
    mgr = DynamicContextManager(window=10, min_execs_at_depth=10, patience=1)
    for _ in range(200):  # build a high peak, then starve it
        mgr.observe(100)
    for _ in range(200):
        mgr.observe(0)
    assert mgr.escalations, "expected at least one escalation"
    assert mgr._peak_rate == 0.0


def test_depth_manager_rejects_invalid_configuration() -> None:
    """Bad ladders and out-of-range parameters fail loudly at construction."""
    with pytest.raises(ValueError):
        DynamicContextManager(ladder=(4, 2, 8))
    with pytest.raises(ValueError):
        DynamicContextManager(start_depth=5)
    with pytest.raises(ValueError):
        DynamicContextManager(saturation_ratio=1.5)
    with pytest.raises(ValueError):
        DynamicContextManager().observe(-1)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_gini_endpoints() -> None:
    """0 for a perfectly equal allocation, near 1 for a maximally concentrated one."""
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-12)
    assert gini([]) == 0.0
    assert gini([0, 0, 0]) == 0.0
    assert gini([0] * 999 + [1000]) > 0.99
    with pytest.raises(ValueError):
        gini([1, -1])


def test_round_robin_has_lower_energy_gini_than_adaptive() -> None:
    """The concentration metric separates the two policies in the stated direction."""
    rr = UniformRoundRobin(random.Random(0), DIMS)
    ad = Adaptive3D(random.Random(0), DIMS)
    for sched in (rr, ad):
        for i in range(30):
            sched.admit(
                _entry(i, novelty={d: (i * 37) % 200 for d in DIMS}, surprisal_bits=i)
            )
        for t in range(600):
            idx, e = sched.select(t)
            sched.report(idx, 0, {}, e)
    assert rr.energy_gini() < ad.energy_gini()


def test_sterile_fraction_counts_only_fuzzed_entries() -> None:
    """Never-selected entries cost queue memory, not budget, and are excluded."""
    sched = UniformRoundRobin(random.Random(0), ("D0",))
    for i in range(4):
        sched.admit(_entry(i))
    idx, e = sched.select(0)
    sched.report(idx, 3, {}, e)  # entry 0 is productive; 1-3 never selected
    assert sched.sterile_fraction() == 0.0


def test_unknown_schedule_name_raises() -> None:
    """A typo in a condition name fails immediately rather than silently."""
    with pytest.raises(KeyError):
        make_schedule("no_such_schedule", random.Random(0), DIMS)


# --------------------------------------------------------------------------- #
# Target invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tgt", get_trigram_targets(), ids=lambda t: t.NAME)
def test_target_is_deterministic(tgt) -> None:
    """The same input always produces the same trace, milestone and verdict."""
    rng = random.Random(9)
    for _ in range(200):
        data = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 40)))
        a, b = tgt.run(data), tgt.run(data)
        assert (a.trace, a.milestone, a.bug) == (b.trace, b.milestone, b.bug)


@pytest.mark.parametrize("tgt", get_trigram_targets(), ids=lambda t: t.NAME)
def test_milestone_never_exceeds_the_declared_maximum(tgt) -> None:
    """``MAX_MILESTONE`` is a real bound, so progress curves are comparable."""
    rng = random.Random(4)
    for _ in range(3000):
        data = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 32)))
        assert 0 <= tgt.run(data).milestone <= tgt.MAX_MILESTONE


@pytest.mark.parametrize("tgt", get_trigram_targets(), ids=lambda t: t.NAME)
def test_bug_implies_the_top_milestone(tgt) -> None:
    """The oracle cannot fire without the ladder being complete."""
    rng = random.Random(6)
    fired = 0
    # A uniform sample never reaches these oracles - that is the whole point of
    # the targets - so the sample is seeded with key-word-derived inputs and
    # then perturbed, which straddles the bug predicate in both directions.
    for i in range(4000):
        if i % 2:
            data = bytes(rng.getrandbits(8) for _ in range(rng.randint(4, 24)))
        else:
            data = bytearray(_solution_input(tgt))
            for _ in range(rng.randint(0, 3)):
                data[rng.randrange(len(data))] = rng.getrandbits(8)
            data = bytes(data)
        res = tgt.run(data)
        if res.bug:
            fired += 1
            assert res.milestone == tgt.MAX_MILESTONE
    assert fired > 0, "no bug fired: the sample never reached the oracle"


@pytest.mark.parametrize("tgt", [TG1TrigramLock(), TG2ValueTrigram()], ids=["TG1", "TG2"])
def test_walk_targets_emit_a_fixed_length_trace_on_the_failing_path(tgt) -> None:
    """Fixed walk length: no hit-count class can leak progress.

    A variable-length walk would hand the edge-coverage baseline a free gradient
    through AFL's saturating count classes, which would invalidate the entire
    comparison. Traces are compared only among non-triggering inputs, since the
    bug block is an extra trace point by design.
    """
    rng = random.Random(2)
    # Compare the WALK portion only. TG2's post-walk tail differs between the
    # phase-one-incomplete path (0x43) and the phase-two path (0x44) by design -
    # that difference is a genuine, intended edge, and it is not partial progress
    # along the ladder. What must not vary is the walk itself.
    walk_lengths = set()
    for _ in range(400):
        data = bytes(rng.getrandbits(8) for _ in range(rng.randint(16, 40)))
        res = tgt.run(data)
        walk = [t for t in res.trace if 0x50 <= t < 0x50 + tgt.ALPHABET]
        walk_lengths.add(len(walk))
    assert walk_lengths == {tgt.WALK}, f"walk length varied: {sorted(walk_lengths)}"


def test_tg1_witness_pair_is_bigram_identical_with_opposite_outcomes() -> None:
    """The core blind-spot claim, checked against the running implementation.

    ``WITNESS_POS`` and ``WITNESS_NEG`` are different Eulerian trails of the same
    transition multigraph. They must agree on the bigram multiset and on every
    per-symbol execution count - which is what makes their AFL maps identical -
    while disagreeing on the trigram multiset and on the bug.
    """
    t = TG1TrigramLock()
    pos, neg = t.WITNESS_POS, t.WITNESS_NEG
    assert bigram_multiset(pos) == bigram_multiset(neg)
    assert sorted(pos) == sorted(neg)  # identical per-symbol counts
    assert trigram_multiset(pos) != trigram_multiset(neg)

    rp = t.run(_encode(pos, t.DECODE_XOR, t.MASK))
    rn = t.run(_encode(neg, t.DECODE_XOR, t.MASK))
    assert rp.bug and rp.milestone == t.MAX_MILESTONE
    assert not rn.bug and rn.milestone < t.MAX_MILESTONE


def test_tg1_witness_pair_produces_an_identical_afl_map() -> None:
    """D0 cannot separate the witness pair, but the 3D engine can.

    This is the operational form of the Step 1 refinement theorem: the pair is in
    one cell of the edge-coverage partition and in two cells of the 3D partition.
    """
    t = TG1TrigramLock()

    def d0_edges(word):
        """Edge multiset of the walk, excluding the post-hoc bug block.

Restricted to edges between two symbol-handler blocks, which is exactly
        the object the theorem is about: the bigram multiset of the walk. The
        post-walk tail is excluded because the bug block executes only *after*
        the ladder is already complete, so the edges around it are a consequence
        of having found the bug rather than a signal that could have led to it.
        Including them would make the comparison vacuous - of course the
        triggering input has an extra edge. The question is whether anything
        *before* the trigger distinguishes the pair, and nothing does.
        """
        eng = MultiDimGuidanceEngine(dims=("D0", "D1"), map_bits=16, context_depth=4)
        eng.reset()
        t.run(_encode(word, t.DECODE_XOR, t.MASK), eng)
        lo, hi = 0x50, 0x50 + t.ALPHABET
        edges = {
            e: c
            for e, c in eng.d0_signature_counted()
            if e[0] is not None and lo <= e[0] < hi and lo <= e[1] < hi
        }
        return frozenset(edges.items()), frozenset(eng.d1.trace_signature())

    pos_d0, pos_d1 = d0_edges(t.WITNESS_POS)
    neg_d0, neg_d1 = d0_edges(t.WITNESS_NEG)
    assert pos_d0 == neg_d0, "AFL maps must be identical up to the bug block"
    assert pos_d1 != neg_d1, "D1 must separate the pair"


def test_tg1_ladder_is_monotone_in_prefix_length() -> None:
    """Each additional key-word symbol never lowers the milestone.

    This is the property that makes the ladder climbable: partial progress is
    retained rather than reset, so a mutation that adds one rung is rewarded.
    """
    t = TG1TrigramLock()
    prev = -1
    for k in range(3, len(t.WITNESS_POS) + 1):
        ms = t.run(_encode(t.WITNESS_POS[:k], t.DECODE_XOR, t.MASK)).milestone
        assert ms >= prev, f"milestone fell at prefix length {k}"
        prev = ms
    assert prev == t.MAX_MILESTONE


def test_tg2_value_gate_is_unreachable_until_the_trigram_ladder_completes() -> None:
    """The two phases are strictly sequential, which is TG2's whole point."""
    t = TG2ValueTrigram()
    magic = t.MAGIC.to_bytes(4, "little")

    bad = bytearray(_encode(t.WITNESS_NEG, t.DECODE_XOR, t.MASK))
    bad[t.VAL_OFF : t.VAL_OFF + 4] = magic
    r_bad = t.run(bytes(bad))
    assert not r_bad.bug
    assert 0x43 in r_bad.trace and 0x44 not in r_bad.trace  # phase two not entered

    good = bytearray(_encode(t.WITNESS_POS, t.DECODE_XOR, t.MASK))
    good[t.VAL_OFF : t.VAL_OFF + 4] = magic
    r_good = t.run(bytes(good))
    assert r_good.bug and 0x44 in r_good.trace


def test_tg2_magic_is_solvable_one_byte_at_a_time() -> None:
    """Each matching magic byte raises the milestone: the D2 byte-split gradient."""
    t = TG2ValueTrigram()
    magic = t.MAGIC.to_bytes(4, "little")
    base = bytearray(_encode(t.WITNESS_POS, t.DECODE_XOR, t.MASK))
    base[t.VAL_OFF : t.VAL_OFF + 4] = bytes(4)
    seen = []
    for k in range(5):
        buf = bytearray(base)
        buf[t.VAL_OFF : t.VAL_OFF + k] = magic[:k]
        seen.append(t.run(bytes(buf)).milestone)
    assert seen == sorted(seen)
    assert seen[-1] == t.MAX_MILESTONE


def test_tg3_chain_advances_one_state_per_correct_op() -> None:
    """The state machine is a genuine chain, and the top state is absorbing."""
    t = TG3SurprisalMaze()
    # The filler byte must decode to an op that advances NO state, or it would
    # extend the prefix by accident: ADVANCE covers ops {0,1,2,3,5,6}, so the
    # filler is chosen to decode to 4.
    fill = next(
        b for b in range(256) if ((b ^ t.DECODE_XOR) & t.OP_MASK) not in t.ADVANCE
    )
    for k in range(len(t.WITNESS_OPS) + 1):
        ops = t.WITNESS_OPS[:k]
        ms = t.run(
            _encode(ops, t.DECODE_XOR, t.OP_MASK, length=16, fill=fill)
        ).milestone
        assert ms == k, f"expected milestone {k} for a {k}-op prefix, got {ms}"


def test_tg3_call_stack_is_shallow_so_d1_depth_is_irrelevant() -> None:
    """TG3 isolates D3: its D1 signature is identical at every window depth.

    Documented as a design property of the target, so it is checked rather than
    assumed - any advantage a 3D condition shows on TG3 is attributable to D2 or
    D3, never to the context window.
    """
    t = TG3SurprisalMaze()
    data = _encode(t.WITNESS_OPS, t.DECODE_XOR, t.OP_MASK, length=16)
    sigs = []
    for depth in DEPTH_LADDER:
        eng = MultiDimGuidanceEngine(dims=("D0", "D1"), map_bits=16, context_depth=depth)
        eng.reset()
        t.run(data, eng)
        sigs.append(frozenset(eng.d1.trace_signature()))
    assert all(s == sigs[0] for s in sigs)


@pytest.mark.parametrize("tgt", get_trigram_targets(), ids=lambda t: t.NAME)
def test_partial_progress_emits_no_extra_trace_point(tgt) -> None:
    """Climbing a rung is invisible in control flow - the blind-spot property.

    Among non-triggering inputs that reach different milestones, the multiset of
    trace points must be indistinguishable in *length*; if a rung emitted a
    block, edge coverage would see partial progress and the benchmark would
    measure nothing.
    """
    rng = random.Random(8)
    by_ms = {}
    for _ in range(4000):
        data = bytes(rng.getrandbits(8) for _ in range(20))
        res = tgt.run(data)
        if res.bug:
            continue
        by_ms.setdefault(res.milestone, set()).add(len(res.trace))
    reached = {m: lens for m, lens in by_ms.items() if m > 0}
    assert len(reached) >= 2, "sample did not reach two distinct milestones"
    all_lengths = set().union(*by_ms.values())
    assert len(all_lengths) == 1, f"trace length leaks progress: {sorted(all_lengths)}"
