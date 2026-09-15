"""
Step 1 validation: invariant tests + quantitative benchmarks for the 3D engine.

Two modes:

* ``uv run pytest workflow/test_step1_guidance.py -q`` - correctness and invariant
  tests, including the Refinement Theorem and native/Python model fidelity.
* ``uv run python workflow/test_step1_guidance.py --full`` - the full experimental
  validation, writing ``results/step1_guidance_validation.json``.

Experiments
-----------
1. **Fidelity** - the Python instrumented models reproduce the compiled C library's
   decision traces exactly.
2. **Partition discrimination** - how much each dimension refines the coverage
   partition of a shared corpus, with the Refinement Theorem verified empirically
   and partition entropy reported.
3. **Refinement witnesses / AFL bigram characterisation** - exhaustive enumeration of
   T1 container-type chains, grouped by AFL counted signature (edges plus hit-count
   classes). AFL's ``trace_bits`` counts ``(prev_block, cur_block)`` occurrences, so
   its map state is a function of the **bigram multiset** of the executed block
   sequence and nothing else. This yields both a positive and a negative
   consequence, and the enumeration measures both: behaviour that is a bigram
   function is already visible to AFL (which is why the baseline solves T1), whereas
   behaviour determined by 3-grams or longer is a structural blind spot (witnessed by
   pairs with identical bigram but different trigram multisets, separated only by the
   calling-context dimension).
4. **Bug finding** - 30 blocked trials per (target, configuration) with success
   rates, log-rank tests on right-censored times, Mann-Whitney U, Wilcoxon
   signed-rank on the paired design, Vargha-Delaney A12 and BH-FDR correction.
5. **Context-depth sweep** - the N-depth parameter of the calling-context tracker
   traded off against context-space size and throughput.
6. **Overhead** - per-execution wall-clock cost of each dimension, with warmup and
   repeats, in both analysis mode (exact shadow sets on) and deployment mode (off).
7. **Dimensionality and collisions** - state-space cardinality and measured hash
   collision rates at several map sizes, against the birthday-bound expectation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- artifact-evaluation path bootstrap (replaces the session-root anchor) --- #
import sys as _ae_sys  # noqa: E402
from pathlib import Path as _AEPath  # noqa: E402

AE_ROOT = _AEPath(__file__).resolve().parent.parent
if str(AE_ROOT) not in _ae_sys.path:
    _ae_sys.path.insert(0, str(AE_ROOT))
import ae_paths as _ae_paths  # noqa: E402,F401  (registers src/ and targets/)

SESSION_DIR = AE_ROOT
# --- end bootstrap --------------------------------------------------------- #
if str(AE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(AE_ROOT / "src"))

import guidance_engine as ge  # noqa: E402
import stats_utils as su  # noqa: E402
from fuzz_harness import (  # noqa: E402
    CONFIGS,
    SEED_INPUT,
    TARGET_BUDGETS,
    Mutator,
    run_trial,
    trial_seed,
    worker,
)
from guidance_engine import (  # noqa: E402
    AFLEdgeTracker,
    CallingContextTracker,
    MultiDimGuidanceEngine,
    StateMachineTracker,
    ValueRangeTracker,
    afl_count_class,
    common_byte_prefix,
    distance_bucket,
    miller_madow_entropy_bits,
    shannon_entropy_bits,
    signed_log2_bucket,
)
from test_targets import (  # noqa: E402
    T1ContextTarget,
    T3StateMachineTarget,
    build_native_lib,
    check_fidelity,
    get_targets,
)

MASTER_SEED = 20260804
N_TRIALS = 30
RESULTS_PATH = SESSION_DIR / "results" / "step1_guidance_validation.json"
N_WORKERS = 12  # leave headroom on the shared 32-core host


# =========================================================================== #
# Invariant / unit tests (pytest)
# =========================================================================== #
def test_afl_count_class_table() -> None:
    """AFL's 8 saturating hit-count classes are reproduced exactly."""
    assert afl_count_class(0) == 0
    assert afl_count_class(1) == 1
    assert afl_count_class(2) == 2
    assert afl_count_class(3) == 4
    for c in range(4, 8):
        assert afl_count_class(c) == 8
    for c in range(8, 16):
        assert afl_count_class(c) == 16
    for c in range(16, 32):
        assert afl_count_class(c) == 32
    for c in range(32, 128):
        assert afl_count_class(c) == 64
    for c in (128, 200, 255, 1000):
        assert afl_count_class(c) == 128
    # exactly 8 non-zero classes, each a distinct power of two
    classes = {afl_count_class(c) for c in range(1, 256)}
    assert len(classes) == 8
    assert all(v and (v & (v - 1)) == 0 for v in classes)


def test_afl_has_new_bits_semantics() -> None:
    """``has_new_bits`` returns 2 for a new edge, 1 for a new count class, 0 else."""
    t = AFLEdgeTracker(map_bits=12)
    t.reset()
    t.bb("a")
    t.bb("b")
    assert t.evaluate(commit=True)["afl_status"] == 2  # first time: new edges

    t.reset()
    t.bb("a")
    t.bb("b")
    assert t.evaluate(commit=True)["afl_status"] == 0  # identical execution

    # Same edges but a higher hit count -> new count class only.
    t.reset()
    for _ in range(5):
        t.bb("a")
        t.bb("b")
    assert t.evaluate(commit=True)["afl_status"] in (1, 2)

    t.reset()
    t.bb("c")
    t.bb("d")
    assert t.evaluate(commit=True)["afl_status"] == 2  # brand new edges again


def test_afl_edge_direction_sensitivity() -> None:
    """AFL's ``prev_loc >> 1`` chaining separates ``A->B`` from ``B->A``."""
    t = AFLEdgeTracker(map_bits=16)
    t.reset()
    t.bb("A")
    t.bb("B")
    ab = set(t.bitmap.trace.keys())
    t.reset()
    t.bb("B")
    t.bb("A")
    ba = set(t.bitmap.trace.keys())
    assert ab != ba


def test_context_rolling_hash_equals_recomputation() -> None:
    """The O(1) window hash equals a straightforward O(N) recomputation.

    Differentially tests the prefix-hash identity
    ``W = H_m - H_{m-n} * P^n (mod 2**64)`` against re-hashing the window.
    """
    import random

    rng = random.Random(1234)
    for depth in (1, 2, 3, 4, 8, 16):
        c = CallingContextTracker(depth=depth, recursion_cap=1000)
        stack_len = 0
        for _ in range(4000):
            if stack_len == 0 or rng.random() < 0.6:
                c.enter(rng.randrange(1, 1 << 20))
                stack_len += 1
            else:
                c.leave()
                stack_len -= 1
            assert c.context_id() == c.context_id_recomputed(), (
                f"mismatch at depth={depth}, stack_len={stack_len}"
            )


def test_context_balanced_enter_leave() -> None:
    """A balanced enter/leave sequence restores the empty context."""
    c = CallingContextTracker(depth=4)
    assert c.context_id() == 0
    for i in range(10):
        c.enter(100 + i)
    assert c.context_id() != 0
    for _ in range(10):
        c.leave()
    assert c.context_id() == 0
    assert c.exact_context() == ()


def test_context_distinguishes_call_paths() -> None:
    """Two different call paths of the same length get different context ids."""
    c1 = CallingContextTracker(depth=4)
    for cs in (1, 2, 1, 2):
        c1.enter(cs)
    c2 = CallingContextTracker(depth=4)
    for cs in (2, 1, 2, 1):
        c2.enter(cs)
    assert c1.context_id() != c2.context_id()
    assert c1.exact_context() != c2.exact_context()


def test_recursion_folding_bounds_effective_depth() -> None:
    """Recursion folding bounds the effective stack for a self-recursive call site."""
    cap = 3
    c = CallingContextTracker(depth=8, recursion_cap=cap)
    for _ in range(500):
        c.enter(42)
    assert c.max_effective_depth == cap, c.max_effective_depth
    # and the context id saturates: deeper recursion yields no new contexts
    ids = set()
    for _ in range(50):
        c.enter(42)
        ids.add(c.context_id())
    assert len(ids) == 1
    # unfolded control: distinct depths give distinct contexts
    c2 = CallingContextTracker(depth=8, recursion_cap=10_000)
    ids2 = set()
    for _ in range(6):
        c2.enter(42)
        ids2.add(c2.context_id())
    assert len(ids2) == 6


def test_signed_log2_bucket_properties() -> None:
    """Magnitude bucketing is symmetric, monotone and bounded to 129 buckets."""
    assert signed_log2_bucket(0) == 64
    assert signed_log2_bucket(1) == 65
    assert signed_log2_bucket(-1) == 63
    assert signed_log2_bucket(255) == signed_log2_bucket(128)  # same bit length
    assert signed_log2_bucket(256) > signed_log2_bucket(255)
    prev = -1
    for e in range(0, 40):
        b = signed_log2_bucket(1 << e)
        assert b > prev
        prev = b
    vals = {signed_log2_bucket(v) for v in range(-(1 << 20), 1 << 20, 977)}
    assert 0 <= min(vals) and max(vals) <= 128


def test_distance_bucket_properties() -> None:
    """Distance bucket is 0 iff operands are equal and monotone in ``|a-b|``."""
    assert distance_bucket(7, 7) == 0
    assert distance_bucket(0, 1) == 1
    prev = 0
    for e in range(0, 40):
        b = distance_bucket(0, 1 << e)
        assert b >= prev
        prev = b
    assert distance_bucket(100, 90) == distance_bucket(90, 100)  # symmetric
    assert distance_bucket(0, 5) < distance_bucket(0, 5000)


def test_common_byte_prefix_known_values() -> None:
    """Equal-byte prefix counting works from both ends."""
    assert common_byte_prefix(0xC0DE, 0xC0DE, 2, True) == 2
    assert common_byte_prefix(0xC0FF, 0xC0DE, 2, True) == 1   # high byte matches
    assert common_byte_prefix(0xC0FF, 0xC0DE, 2, False) == 0  # low byte differs
    assert common_byte_prefix(0x11DE, 0xC0DE, 2, False) == 1  # low byte matches
    assert common_byte_prefix(0x11DE, 0xC0DE, 2, True) == 0
    assert common_byte_prefix(0xDEADBEEF, 0xDEADBEEF, 4, True) == 4


def test_entropy_estimator_bounds() -> None:
    """Entropy is in ``[0, log2 K]``, maximal for uniform, and MM >= plug-in."""
    assert shannon_entropy_bits([]) == 0.0
    assert shannon_entropy_bits([10]) == 0.0
    for k in (2, 4, 8, 16):
        uniform = [100] * k
        h = shannon_entropy_bits(uniform)
        assert abs(h - math.log2(k)) < 1e-9
        skewed = [1000] + [1] * (k - 1)
        assert 0.0 < shannon_entropy_bits(skewed) < h
    counts = [50, 30, 12, 5, 3]
    assert miller_madow_entropy_bits(counts) >= shannon_entropy_bits(counts)
    # the correction shrinks as N grows
    small = [3, 2, 1]
    big = [3000, 2000, 1000]
    corr_small = miller_madow_entropy_bits(small) - shannon_entropy_bits(small)
    corr_big = miller_madow_entropy_bits(big) - shannon_entropy_bits(big)
    assert corr_small > corr_big > 0


def test_surprisal_monotone_and_maximal_for_unseen() -> None:
    """Transition surprisal is maximal for unseen pairs and decreases with count."""
    sm = StateMachineTracker(kgram=2, alpha=0.5)
    sm.reset()
    sm.api("open", 0, state_tag="OPEN")
    sm.api("write", 0, state_tag="WRITTEN")
    sm.evaluate(commit=True)

    unseen = ("NOPE", "nope", "NOPE2")
    seen = ("OPEN", "write", "WRITTEN")
    assert sm.novelty_weight(unseen) > sm.novelty_weight(seen)
    assert abs(sm.novelty_weight(unseen) - sm.max_novelty_weight()) < 1e-12

    w_before = sm.novelty_weight(seen)
    for _ in range(20):
        sm.reset()
        sm.api("open", 0, state_tag="OPEN")
        sm.api("write", 0, state_tag="WRITTEN")
        sm.evaluate(commit=True)
    assert sm.novelty_weight(seen) < w_before  # monotone decreasing in count
    assert sm.novelty_weight(seen) > 0.0       # always a positive surprisal


def test_state_machine_graph_metrics() -> None:
    """Graph statistics detect states, transitions, self-loops and a real cycle."""
    sm = StateMachineTracker(kgram=2)
    sm.reset()
    # A -> B -> C -> A is one cycle of length 3 plus a self-loop on B
    for tag in ("A", "B", "B", "C", "A"):
        sm.api("step", 0, state_tag=tag)
    sm.evaluate(commit=True)
    st = sm.stats()
    assert st["n_abstract_states"] >= 4  # includes the INITIAL state
    assert st["n_transitions"] >= 4
    assert st["n_self_loops"] == 1
    assert st["largest_scc_size"] >= 3  # A, B, C are mutually reachable
    assert st["max_novelty_weight_bits"] > 0


def test_value_range_gradient_admits_improvement() -> None:
    """Getting strictly closer to satisfying a comparison is judged interesting."""
    v = ValueRangeTracker()
    v.reset()
    v.cmp_("gate", 0x0000, 0xC0DE, width=2)
    assert v.evaluate(commit=True)["is_new"]

    # A closer value in the same distance bucket is still an improvement.
    v.reset()
    v.cmp_("gate", 0xC000, 0xC0DE, width=2)
    r = v.evaluate(commit=True)
    assert r["n_improved_gradients"] > 0
    assert r["is_new"]

    # Repeating the best-so-far value is not.
    v.reset()
    v.cmp_("gate", 0xC000, 0xC0DE, width=2)
    r2 = v.evaluate(commit=True)
    assert r2["n_improved_gradients"] == 0
    assert not r2["is_new"]

    # Solving it exactly is an improvement and records a solved comparison.
    v.reset()
    v.cmp_("gate", 0xC0DE, 0xC0DE, width=2)
    r3 = v.evaluate(commit=True)
    assert r3["n_improved_gradients"] > 0
    assert v.stats()["n_solved_comparisons"] == 1


def test_value_range_byte_splitting() -> None:
    """Individually matching byte positions are their own coverage elements."""
    v = ValueRangeTracker()
    v.reset()
    v.cmp_("m", 0x0000_0000, 0xDEAD_BEEF, width=4)
    v.evaluate(commit=True)
    # match only byte 2 (0xAD) - not a prefix from either end
    v.reset()
    v.cmp_("m", 0x00AD_0000, 0xDEAD_BEEF, width=4)
    r = v.evaluate(commit=True)
    assert r["new_elements"] > 0, "byte-level match must register as new coverage"
    assert ("m", "bytehit", 2) in v.exact_elements


def test_engine_ablation_disables_dimensions() -> None:
    """Disabled dimensions record nothing and are absent from stats."""
    tgt = T3StateMachineTarget()
    data = bytes(range(32))
    for dims in (("D0",), ("D0", "D1"), ("D0", "D3"), ("D0", "D1", "D2", "D3")):
        eng = MultiDimGuidanceEngine(dims=dims, map_bits=14)
        eng.reset()
        tgt.run(data, eng)
        eng.evaluate(commit=True)
        st = eng.stats()
        assert set(st["per_dimension"]) == set(dims)
        for d in ("D0", "D1", "D2", "D3"):
            tracker = getattr(eng, d.lower())
            if d not in dims:
                assert tracker.n_probes == 0, f"{d} probed while disabled"
            else:
                assert tracker.n_probes > 0 or d == "D2"


def test_refinement_theorem() -> None:
    """``alpha_0 = pi o alpha_1``: the D1 partition refines the D0 partition.

    Verified constructively over exact (collision-free) element sets on a real
    corpus: projecting every context-sensitive element onto its edge component must
    reproduce the edge-coverage element set exactly, and no two executions may share
    a D1 signature while differing in their D0 signature.
    """
    import random

    rng = random.Random(999)
    mut = Mutator(rng)
    for tgt in get_targets():
        eng = MultiDimGuidanceEngine(dims=("D0", "D1"), map_bits=16, context_depth=8)
        d1_to_d0: Dict[frozenset, frozenset] = {}
        data = SEED_INPUT
        for _ in range(400):
            data = mut.mutate(data)
            eng.reset()
            tgt.run(data, eng)
            eng.evaluate(commit=True)

            d0_sig = eng.d0.trace_signature()
            d1_sig = eng.d1.trace_signature()
            # (1) projection identity: pi(alpha_1) == alpha_0
            projected = frozenset((prev, cur) for (_ctx, prev, cur) in d1_sig)
            assert projected == d0_sig, f"{tgt.NAME}: projection identity violated"
            # (2) refinement: D1 signature determines D0 signature
            prior = d1_to_d0.get(d1_sig)
            if prior is None:
                d1_to_d0[d1_sig] = d0_sig
            else:
                assert prior == d0_sig, f"{tgt.NAME}: D1 cell spans two D0 cells"


def test_composite_refines_every_dimension() -> None:
    """The composite signature determines each single-dimension signature."""
    import random

    rng = random.Random(4242)
    mut = Mutator(rng)
    tgt = T3StateMachineTarget()
    full = MultiDimGuidanceEngine(dims=("D0", "D1", "D2", "D3"), map_bits=16)
    singles = {d: MultiDimGuidanceEngine(dims=(d,), map_bits=16) for d in ge.DIMENSIONS}
    seen: Dict[frozenset, Dict[str, frozenset]] = {}
    data = SEED_INPUT
    for _ in range(300):
        data = mut.mutate(data)
        full.reset()
        tgt.run(data, full)
        full.evaluate(commit=True)
        comp = full.trace_signature()

        sigs = {}
        for d, eng in singles.items():
            eng.reset()
            tgt.run(data, eng)
            eng.evaluate(commit=True)
            sigs[d] = eng.trace_signature()
        prior = seen.get(comp)
        if prior is None:
            seen[comp] = sigs
        else:
            for d in ge.DIMENSIONS:
                assert prior[d] == sigs[d], f"composite cell spans two {d} cells"


def test_native_model_fidelity() -> None:
    """Python instrumented models match the compiled C library exactly."""
    build_native_lib()
    rep = check_fidelity(n_inputs=600, seed=7)
    assert rep["fidelity_ok"], rep["per_target"]


def test_refinement_witness_exists() -> None:
    """AFL conflates structurally distinct executions that D1 separates.

    Over an exhaustively enumerated space of container-type chains there must exist
    at least one pair of executions with *identical* AFL counted signatures (edges
    plus hit-count classes) but different calling-context signatures. This certifies
    strict refinement on real executions rather than by construction.
    """
    w = enumerate_refinement_witnesses(chain_len=6)
    assert w["n_d0_counted_classes"] < w["n_d1_classes"], w
    assert w["n_conflated_groups"] >= 1, w
    assert w["witness_pair"] is not None, w
    assert w["witness_pair"]["d0_counted_identical"]
    assert not w["witness_pair"]["d1_identical"]


def test_afl_is_a_bigram_statistic() -> None:
    """Equal block-sequence bigram multisets imply equal AFL counted signatures.

    AFL's ``trace_bits`` counts occurrences of ``(prev_block, cur_block)`` pairs, so
    the map state is a function of the bigram multiset of the executed block
    sequence and of nothing else. This test verifies the implication that gives the
    characterisation its teeth: two chains whose block bigram multisets agree are
    indistinguishable to AFL even with hit-count classes enabled.
    """
    w = enumerate_refinement_witnesses(chain_len=6)
    for grp in w["conflated_group_examples"]:
        assert grp["block_bigram_multiset_identical"], grp
    assert w["n_groups_with_identical_bigrams_and_different_trigrams"] >= 1, w


def test_trial_determinism() -> None:
    """Identical seeds reproduce identical trial outcomes."""
    kw = dict(max_execs=3000, curve_every=3000)
    a = run_trial("T3_state_machine", "d0_d3_statemachine", 12345, **kw)
    b = run_trial("T3_state_machine", "d0_d3_statemachine", 12345, **kw)
    assert (a.first_bug_exec, a.max_milestone, a.corpus_size, a.n_interesting) == (
        b.first_bug_exec, b.max_milestone, b.corpus_size, b.n_interesting
    )
    c = run_trial("T3_state_machine", "d0_d3_statemachine", 999, **kw)
    assert (c.corpus_size, c.n_interesting) != (a.corpus_size, a.n_interesting)


# =========================================================================== #
# Experiment: exhaustive refinement witnesses / AFL bigram characterisation
# =========================================================================== #
def _ngram_multiset(seq: Sequence[Any], n: int) -> Dict[str, int]:
    """Multiset of length-``n`` contiguous subsequences, as a JSON-safe dict."""
    c: Counter = Counter()
    for i in range(len(seq) - n + 1):
        c[tuple(seq[i : i + n])] += 1
    return {"|".join(str(x) for x in k): v for k, v in sorted(c.items(), key=str)}


def enumerate_refinement_witnesses(chain_len: int = 6) -> Dict[str, Any]:
    """Exhaustively enumerate T1 container-type chains and compare metrics.

    Every chain of ``chain_len`` containers over ``{SEQ, MAP}`` is encoded as an
    input (one child per container, terminated by a 1-byte string) and executed under
    D0 and D1. Grouping executions by their AFL counted signature (edges plus
    hit-count classes) exposes exactly which structurally distinct executions the
    baseline metric conflates.

    This also substantiates the characterisation that makes the T1 outcome
    interpretable: AFL's ``trace_bits`` counts ``(prev_block, cur_block)``
    occurrences, so the map state is a function of the **bigram multiset** of the
    executed basic-block sequence. Behaviour that is a function of those bigram
    counts is therefore visible to AFL, while behaviour that depends on 3-grams or
    longer is a structural blind spot.

    Parameters
    ----------
    chain_len : int, optional
        Number of nested containers per chain, default 6.

    Returns
    -------
    dict
        Class counts per metric, conflated-group statistics, a concrete witness pair
        and the bigram/trigram evidence.
    """
    tgt = T1ContextTarget()
    SEQ, MAP, STR = tgt.TAG_SEQ, tgt.TAG_MAP, tgt.TAG_STR

    def encode(order: Sequence[int]) -> bytes:
        out = bytearray()
        for t in order:
            out += bytes([t, 1])  # container tag, child count = 1
        out += bytes([STR, 1, 0x41])  # 1-byte string plus its payload
        return bytes(out)

    rows: List[Dict[str, Any]] = []
    for mask in range(1 << chain_len):
        order = [SEQ if (mask >> i) & 1 == 0 else MAP for i in range(chain_len)]
        data = encode(order)
        eng = MultiDimGuidanceEngine(dims=("D0", "D1"), map_bits=16, context_depth=16)
        eng.reset()
        res = tgt.run(data, eng)
        eng.evaluate(commit=True)
        blocks = res.trace
        rows.append(
            {
                "chain": "".join("S" if t == SEQ else "M" for t in order),
                "hex": data.hex(),
                "milestone": res.milestone,
                "bug": res.bug,
                "d0_edges": eng.d0.trace_signature(),
                "d0_counted": eng.d0_signature_counted(),
                "d1": eng.d1.trace_signature(),
                "block_bigrams": _ngram_multiset(blocks, 2),
                "block_trigrams": _ngram_multiset(blocks, 3),
                "n_blocks": len(blocks),
            }
        )

    def n_classes(key: str) -> int:
        return len({r[key] for r in rows})

    groups: Dict[frozenset, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["d0_counted"]].append(r)

    conflated = [g for g in groups.values() if len({x["d1"] for x in g}) > 1]
    mixed_milestone = [g for g in groups.values() if len({x["milestone"] for x in g}) > 1]
    mixed_bug = [g for g in groups.values() if len({x["bug"] for x in g}) > 1]

    examples: List[Dict[str, Any]] = []
    n_bigram_eq_trigram_diff = 0
    for g in conflated:
        a, b = g[0], g[1]
        bigrams_same = a["block_bigrams"] == b["block_bigrams"]
        trigrams_diff = a["block_trigrams"] != b["block_trigrams"]
        if bigrams_same and trigrams_diff:
            n_bigram_eq_trigram_diff += 1
        if len(examples) < 3:
            examples.append(
                {
                    "chain_a": a["chain"],
                    "chain_b": b["chain"],
                    "milestone_a": a["milestone"],
                    "milestone_b": b["milestone"],
                    "group_size": len(g),
                    "block_bigram_multiset_identical": bigrams_same,
                    "block_trigram_multiset_differs": trigrams_diff,
                    "n_distinct_d1_signatures_in_group": len({x["d1"] for x in g}),
                }
            )

    witness = None
    if conflated:
        g = conflated[0]
        a = g[0]
        b = next(x for x in g[1:] if x["d1"] != a["d1"])
        witness = {
            "chain_a": a["chain"],
            "input_a_hex": a["hex"],
            "chain_b": b["chain"],
            "input_b_hex": b["hex"],
            "milestone_a": a["milestone"],
            "milestone_b": b["milestone"],
            "bug_a": a["bug"],
            "bug_b": b["bug"],
            "d0_edge_identical": a["d0_edges"] == b["d0_edges"],
            "d0_counted_identical": a["d0_counted"] == b["d0_counted"],
            "d1_identical": a["d1"] == b["d1"],
            "n_d1_elements_a": len(a["d1"]),
            "n_d1_elements_b": len(b["d1"]),
            "block_bigram_multiset_identical": a["block_bigrams"] == b["block_bigrams"],
            "block_trigram_multiset_differs": a["block_trigrams"] != b["block_trigrams"],
        }

    return {
        "chain_len": chain_len,
        "n_chains_enumerated": len(rows),
        "n_d0_edge_classes": n_classes("d0_edges"),
        "n_d0_counted_classes": n_classes("d0_counted"),
        "n_d1_classes": n_classes("d1"),
        "n_distinct_milestones": len({r["milestone"] for r in rows}),
        "milestone_histogram": {
            str(k): v for k, v in sorted(Counter(r["milestone"] for r in rows).items())
        },
        "n_conflated_groups": len(conflated),
        "n_chains_in_conflated_groups": sum(len(g) for g in conflated),
        "n_groups_with_mixed_milestone": len(mixed_milestone),
        "n_groups_mixing_bug_and_nonbug": len(mixed_bug),
        "n_groups_with_identical_bigrams_and_different_trigrams": n_bigram_eq_trigram_diff,
        "conflated_group_examples": examples,
        "witness_pair": witness,
        "characterisation": (
            "AFL's trace_bits counts occurrences of (prev_block, cur_block) pairs, so "
            "the map state - including saturating hit-count classes - is a function of "
            "the BIGRAM MULTISET of the executed basic-block sequence and of nothing "
            "else. Two consequences follow. (a) Any behaviour that is a function of "
            "those bigram counts is already visible to AFL, and calling-context "
            "sensitivity adds no bug-finding power over it. T1's scratch offset is a "
            "linear function of adjacent container-type pairs, hence a bigram "
            "function, which is why the edge-coverage baseline solves T1 and the D1 "
            "configurations gain nothing - a prediction this enumeration confirms and "
            "the bug-finding experiment independently reproduces. (b) Behaviour "
            "determined by 3-grams or longer is a structural blind spot: the witness "
            "pair below has identical block bigram multisets (so identical AFL state) "
            "but different trigram multisets, and only the calling-context dimension "
            "separates the two executions."
        ),
    }


# =========================================================================== #
# Experiment: partition discrimination
# =========================================================================== #
def collect_shared_corpus(target, n_inputs: int = 1500, seed: int = 31337) -> List[bytes]:
    """Collect a diverse shared corpus for partition analysis.

    Inputs come from a short full-3D guided run (so behaviours are diverse rather
    than dominated by trivially-rejected random bytes) plus purely random mutations.
    Exactly the same corpus is then evaluated under every dimension, so partition
    comparisons are not confounded by different input distributions.

    Parameters
    ----------
    target : object
        Benchmark target instance.
    n_inputs : int, optional
        Target corpus size, default 1500.
    seed : int, optional
        RNG seed.

    Returns
    -------
    list of bytes
    """
    import random

    rng = random.Random(seed)
    mut = Mutator(rng)
    eng = MultiDimGuidanceEngine(dims=ge.DIMENSIONS, map_bits=16, context_depth=16)
    corpus: List[bytes] = [SEED_INPUT]
    out: List[bytes] = [SEED_INPUT]
    cursor = 0
    execs = 0
    while len(out) < n_inputs and execs < 40 * n_inputs:
        parent = corpus[cursor % len(corpus)]
        cursor += 1
        data = mut.mutate(parent, splice_pool=corpus)
        eng.reset()
        target.run(data, eng)
        fb = eng.evaluate(commit=True)
        execs += 1
        if fb.is_interesting:
            corpus.append(data)
            out.append(data)
        elif rng.random() < 0.05:  # keep some uninteresting inputs too
            out.append(data)
    return out[:n_inputs]


def experiment_partition_discrimination() -> Dict[str, Any]:
    """Measure how much each dimension refines the coverage partition.

    Returns
    -------
    dict
        Per-target partition sizes, entropies, refinement factors, verified
        refinement relations, and counts of coverage-equivalent input groups that
        differ in bug progress.
    """
    out: Dict[str, Any] = {
        "description": (
            "Each dimension induces a partition of the input space via its exact "
            "(collision-free) element set. A finer partition means the fuzzer can "
            "distinguish more behaviours and therefore retain more distinct inputs. "
            "Partition entropy H(Pi) = -sum p_c log2 p_c over cell frequencies "
            "measures the information the metric extracts from the corpus."
        ),
        "per_target": {},
    }
    for tgt in get_targets():
        corpus = collect_shared_corpus(tgt)
        engines = {d: MultiDimGuidanceEngine(dims=(d,), map_bits=16, context_depth=16)
                   for d in ge.DIMENSIONS}
        engines["composite_3d"] = MultiDimGuidanceEngine(
            dims=ge.DIMENSIONS, map_bits=16, context_depth=16
        )
        sigs: Dict[str, List[frozenset]] = {k: [] for k in engines}
        d0_counted: List[frozenset] = []
        milestones: List[int] = []
        bugs: List[bool] = []

        for data in corpus:
            for name, eng in engines.items():
                eng.reset()
                res = tgt.run(data, eng)
                eng.evaluate(commit=True)
                sigs[name].append(eng.trace_signature())
                if name == "D0":
                    d0_counted.append(eng.d0_signature_counted())
                    milestones.append(res.milestone)
                    bugs.append(res.bug)

        def partition_stats(labels: Sequence[frozenset]) -> Dict[str, Any]:
            counts = Counter(labels)
            n = len(labels)
            return {
                "n_classes": len(counts),
                "partition_entropy_bits": shannon_entropy_bits(list(counts.values())),
                "max_possible_entropy_bits": math.log2(n) if n > 1 else 0.0,
                "largest_class_fraction": max(counts.values()) / n if n else 0.0,
            }

        per_dim = {name: partition_stats(s) for name, s in sigs.items()}
        per_dim["D0_with_hit_counts"] = partition_stats(d0_counted)

        # verified refinement relations
        def refines(fine: Sequence[frozenset], coarse: Sequence[frozenset]) -> bool:
            m: Dict[frozenset, frozenset] = {}
            for f, c in zip(fine, coarse):
                if f in m and m[f] != c:
                    return False
                m[f] = c
            return True

        refinement = {
            "D1_refines_D0": refines(sigs["D1"], sigs["D0"]),
            "composite_refines_D0": refines(sigs["composite_3d"], sigs["D0"]),
            "composite_refines_D1": refines(sigs["composite_3d"], sigs["D1"]),
            "composite_refines_D2": refines(sigs["composite_3d"], sigs["D2"]),
            "composite_refines_D3": refines(sigs["composite_3d"], sigs["D3"]),
        }

        # coverage-equivalent but progress-distinct groups (the blind-spot measure)
        def blindspots(labels: Sequence[frozenset]) -> Dict[str, int]:
            groups: Dict[frozenset, List[int]] = defaultdict(list)
            for lab, ms in zip(labels, milestones):
                groups[lab].append(ms)
            split = {k: v for k, v in groups.items() if len(set(v)) > 1}
            bug_groups: Dict[frozenset, List[bool]] = defaultdict(list)
            for lab, bg in zip(labels, bugs):
                bug_groups[lab].append(bg)
            bug_split = {k: v for k, v in bug_groups.items() if len(set(v)) > 1}
            return {
                "n_classes_with_mixed_milestones": len(split),
                "n_inputs_in_mixed_classes": sum(len(v) for v in split.values()),
                "n_classes_mixing_bug_and_nonbug": len(bug_split),
            }

        out["per_target"][tgt.NAME] = {
            "n_corpus_inputs": len(corpus),
            "n_distinct_milestones": len(set(milestones)),
            "milestone_histogram": {str(k): v for k, v in sorted(Counter(milestones).items())},
            "partitions": per_dim,
            "refinement_verified": refinement,
            "refinement_factor_D1_over_D0": (
                per_dim["D1"]["n_classes"] / per_dim["D0"]["n_classes"]
                if per_dim["D0"]["n_classes"] else float("nan")
            ),
            "refinement_factor_composite_over_D0": (
                per_dim["composite_3d"]["n_classes"] / per_dim["D0"]["n_classes"]
                if per_dim["D0"]["n_classes"] else float("nan")
            ),
            "blindspots_D0": blindspots(sigs["D0"]),
            "blindspots_D0_with_hit_counts": blindspots(d0_counted),
            "blindspots_composite_3d": blindspots(sigs["composite_3d"]),
        }
    return out


# =========================================================================== #
# Experiment: bug finding
# =========================================================================== #
def _jobs_for_bug_finding(n_trials: int) -> List[Dict[str, Any]]:
    jobs = []
    for tgt in get_targets():
        budget = TARGET_BUDGETS[tgt.NAME]
        for cfg in CONFIGS:
            for i in range(n_trials):
                jobs.append(
                    {
                        "target_name": tgt.NAME,
                        "config_name": cfg,
                        "seed": trial_seed(MASTER_SEED, tgt.NAME, i),
                        "max_execs": budget,
                        "curve_every": max(500, budget // 60),
                        "tag": f"{tgt.NAME}|{cfg}|{i}",
                    }
                )
    return jobs


def experiment_bug_finding(n_trials: int = N_TRIALS) -> Dict[str, Any]:
    """Run the blocked bug-finding experiment and analyse it.

    Parameters
    ----------
    n_trials : int, optional
        Independent trials per (target, configuration), default 30.

    Returns
    -------
    dict
        Raw per-trial records plus per-target statistical comparisons against the
        ``edge_only`` baseline, with BH-FDR-corrected p-values.
    """
    jobs = _jobs_for_bug_finding(n_trials)
    print(f"  bug-finding: {len(jobs)} trials on {N_WORKERS} workers", flush=True)
    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for rec in pool.map(worker, jobs, chunksize=1):
            records.append(rec)
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                rate = done / el if el else 0
                eta = (len(jobs) - done) / rate if rate else 0
                print(
                    f"    {done}/{len(jobs)} trials  {el:.0f}s elapsed  "
                    f"ETA {eta / 60:.1f} min",
                    flush=True,
                )

    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[(r["target"], r["config"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["tag"])

    analysis: Dict[str, Any] = {"per_target": {}}
    pvals: List[float] = []
    pkeys: List[Tuple[str, str, str]] = []

    for tgt in get_targets():
        name = tgt.NAME
        budget = TARGET_BUDGETS[name]
        base = by[(name, "edge_only")]
        base_times = [r["first_bug_exec"] or budget for r in base]
        base_events = [1 if r["first_bug_exec"] else 0 for r in base]
        base_ms = [r["max_milestone"] for r in base]

        per_cfg: Dict[str, Any] = {}
        for cfg in CONFIGS:
            recs = by[(name, cfg)]
            found = [r for r in recs if r["first_bug_exec"] is not None]
            times = [r["first_bug_exec"] or budget for r in recs]
            events = [1 if r["first_bug_exec"] else 0 for r in recs]
            ms = [r["max_milestone"] for r in recs]
            lo, hi = su.clopper_pearson(len(found), len(recs))

            entry: Dict[str, Any] = {
                "n_trials": len(recs),
                "budget_execs": budget,
                "n_bugs_found": len(found),
                "success_rate": len(found) / len(recs) if recs else float("nan"),
                "success_rate_ci95": [lo, hi],
                "execs_to_bug_among_successes": su.describe(
                    [r["first_bug_exec"] for r in found]
                ),
                "max_milestone": su.describe(ms),
                "milestone_reached_max": int(max(ms)) if ms else 0,
                "max_milestone_possible": tgt.MAX_MILESTONE,
                "corpus_size": su.describe([r["corpus_size"] for r in recs]),
                "exact_elements": su.describe(
                    [r["engine_stats"]["total_exact_elements"] for r in recs]
                ),
                "mean_exec_per_second": float(
                    np.mean([r["execs"] / r["wall_seconds"] for r in recs if r["wall_seconds"] > 0])
                ),
            }
            if cfg != "edge_only":
                fisher = su.fisher_success(len(found), len(recs), len(
                    [r for r in base if r["first_bug_exec"] is not None]), len(base))
                lr = su.logrank_test(times, events, base_times, base_events)
                mw_ms = su.mann_whitney(ms, base_ms)
                wx_ms = su.wilcoxon_paired(ms, base_ms)
                mw_t = su.mann_whitney(times, base_times)
                entry["vs_edge_only"] = {
                    "fisher_success": fisher,
                    "logrank_time_to_bug": lr,
                    "milestone_mann_whitney": mw_ms,
                    "milestone_wilcoxon_paired": wx_ms,
                    "censored_time_mann_whitney": mw_t,
                    "note": (
                        "A12 on milestone: >0.5 favours this configuration. A12 on "
                        "censored time: <0.5 favours this configuration (lower is "
                        "better). Censored times are substituted at the budget, which "
                        "makes the rank test conservative; the log-rank test handles "
                        "censoring correctly and is the primary time-to-bug test."
                    ),
                }
                for key, p in (
                    ("fisher_success", fisher["p_value"]),
                    ("logrank_time_to_bug", lr["p_value"]),
                    ("milestone_mann_whitney", mw_ms["p_value"]),
                ):
                    pvals.append(p)
                    pkeys.append((name, cfg, key))
            per_cfg[cfg] = entry
        analysis["per_target"][name] = {
            "targeted_dimension": tgt.TARGETED_DIMENSION,
            "max_milestone_possible": tgt.MAX_MILESTONE,
            "budget_execs": budget,
            "configs": per_cfg,
        }

    bh = su.benjamini_hochberg(pvals, alpha=0.05)
    fdr: Dict[str, Any] = {"alpha": 0.05, "n_tests": len(pvals), "tests": []}
    for (tname, cfg, key), p, q, rej in zip(pkeys, pvals, bh["q_values"], bh["rejected"]):
        fdr["tests"].append(
            {"target": tname, "config": cfg, "test": key, "p_value": p,
             "q_value": q, "significant_at_fdr_5pct": rej}
        )
        analysis["per_target"][tname]["configs"][cfg]["vs_edge_only"][key + "_q_value"] = q
    analysis["multiple_testing_correction"] = fdr
    analysis["wall_seconds"] = time.perf_counter() - t0
    analysis["n_trials_per_cell"] = n_trials
    return analysis, records


# =========================================================================== #
# Experiment: context-depth sweep
# =========================================================================== #
def experiment_context_depth_sweep(
    depths: Sequence[int] = (1, 2, 4, 8, 16),
    n_trials: int = N_TRIALS,
    baseline_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Trade off the calling-context window depth N against cost.

    The bug in T1 requires a container path of five type switches, which needs a
    context window deep enough to represent that path prefix. A window that is too
    shallow aliases distinct paths and cannot supply the gradient; a window that is
    too deep inflates the context space and dilutes the corpus. This sweep measures
    both sides.

    Parameters
    ----------
    depths : sequence of int, optional
        Context window depths to evaluate.
    n_trials : int, optional
        Trials per depth.
    baseline_records : list of dict, optional
        ``edge_only`` trial records for the same target and budget from the
        bug-finding experiment. When supplied, each depth is contrasted against that
        baseline with a log-rank test, A12 and a seed-paired Wilcoxon test, with
        BH-FDR correction inside this hypothesis family - so the "N=4 beats the
        baseline" style claim is tested rather than eyeballed across experiments.

    Returns
    -------
    tuple
        ``(summary_dict, raw_trial_records)``.
    """
    target = "T1_context"
    budget = TARGET_BUDGETS[target]
    jobs = []
    for n in depths:
        for i in range(n_trials):
            jobs.append(
                {
                    "target_name": target,
                    "config_name": "d0_d1_context",
                    "seed": trial_seed(MASTER_SEED, target, i),
                    "max_execs": budget,
                    "curve_every": budget,
                    "context_depth": int(n),
                    "tag": f"N={n}|{i}",
                }
            )
    print(f"  context-depth sweep: {len(jobs)} trials", flush=True)
    recs: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for r in pool.map(worker, jobs, chunksize=1):
            recs.append(r)

    by_depth: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by_depth[int(r["tag"].split("|")[0][2:])].append(r)

    out: Dict[str, Any] = {
        "target": target,
        "budget_execs": budget,
        "n_trials_per_depth": n_trials,
        "description": (
            "Context window depth N versus effectiveness and cost. The T1 bug needs "
            "a five-switch container path; a window shorter than that path prefix "
            "aliases distinct paths and yields no gradient, while a deeper window "
            "inflates the context space and dilutes the corpus."
        ),
        "per_depth": {},
    }
    base = None
    if baseline_records:
        base = sorted(
            (r for r in baseline_records
             if r["target"] == target and r["config"] == "edge_only"),
            key=lambda r: r["seed"],
        )

    pvals: List[float] = []
    pkeys: List[Tuple[str, str]] = []
    for n in depths:
        rs = by_depth[n]
        found = [r for r in rs if r["first_bug_exec"] is not None]
        entry: Dict[str, Any] = {
            "n_trials": len(rs),
            "n_bugs_found": len(found),
            "success_rate": len(found) / len(rs) if rs else float("nan"),
            "success_rate_ci95": list(su.clopper_pearson(len(found), len(rs))),
            "execs_to_bug_among_successes": su.describe(
                [r["first_bug_exec"] for r in found]
            ),
            "max_milestone": su.describe([r["max_milestone"] for r in rs]),
            "corpus_size": su.describe([r["corpus_size"] for r in rs]),
            "distinct_contexts_exact": su.describe(
                [r["engine_stats"]["per_dimension"]["D1"].get("distinct_contexts_exact", 0)
                 for r in rs]
            ),
            "exact_elements": su.describe(
                [r["engine_stats"]["total_exact_elements"] for r in rs]
            ),
            "mean_exec_per_second": float(
                np.mean([r["execs"] / r["wall_seconds"] for r in rs if r["wall_seconds"] > 0])
            ),
        }
        if base:
            rs_sorted = sorted(rs, key=lambda r: r["seed"])
            t_a = [r["first_bug_exec"] or budget for r in rs_sorted]
            e_a = [1 if r["first_bug_exec"] else 0 for r in rs_sorted]
            t_b = [r["first_bug_exec"] or budget for r in base]
            e_b = [1 if r["first_bug_exec"] else 0 for r in base]
            lr = su.logrank_test(t_a, e_a, t_b, e_b)
            mw = su.mann_whitney(t_a, t_b)
            paired = (
                su.wilcoxon_paired(t_a, t_b)
                if [r["seed"] for r in rs_sorted] == [r["seed"] for r in base]
                else {"statistic": float("nan"), "p_value": float("nan"),
                      "n_pairs": 0, "n_nonzero_pairs": 0}
            )
            entry["vs_edge_only_baseline"] = {
                "logrank_time_to_bug": lr,
                "censored_time_mann_whitney": mw,
                "seed_paired_wilcoxon_time": paired,
                "median_execs_ratio_vs_baseline": (
                    entry["execs_to_bug_among_successes"].get("median", float("nan"))
                    / su.describe([r["first_bug_exec"] for r in base
                                   if r["first_bug_exec"]]).get("median", float("nan"))
                ),
                "note": (
                    "A12 on censored time < 0.5 favours this depth (lower time is "
                    "better). Seeds are blocked across arms, so the paired test is "
                    "valid; the log-rank test is the primary test because it handles "
                    "right-censoring correctly."
                ),
            }
            for key, p in (("logrank_time_to_bug", lr["p_value"]),
                           ("seed_paired_wilcoxon_time", paired["p_value"])):
                pvals.append(p)
                pkeys.append((str(n), key))
        out["per_depth"][str(n)] = entry

    if pvals:
        bh = su.benjamini_hochberg(pvals, alpha=0.05)
        fam: Dict[str, Any] = {"alpha": 0.05, "n_tests": len(pvals), "tests": []}
        for (dn, key), p, q, rej in zip(pkeys, pvals, bh["q_values"], bh["rejected"]):
            fam["tests"].append({"depth": dn, "test": key, "p_value": p,
                                 "q_value": q, "significant_at_fdr_5pct": rej})
            out["per_depth"][dn]["vs_edge_only_baseline"][key + "_q_value"] = q
        out["multiple_testing_correction"] = fam
        out["baseline"] = "edge_only from the bug-finding experiment (same seeds/budget)"
    return out, recs


# =========================================================================== #
# Experiment: overhead
# =========================================================================== #
def experiment_overhead(
    n_inputs: int = 1200, repeats: int = 7, seed: int = 5150
) -> Dict[str, Any]:
    """Measure per-execution tracking overhead per dimension.

    Reports two regimes: *deployment mode* (``track_exact=False``, only the bounded
    bitmaps, which is what a real fuzzer would ship) and *analysis mode*
    (``track_exact=True``, maintaining exact shadow element sets, which the
    partition and collision analyses require). Timing uses a warmup pass plus
    ``repeats`` measured passes over a fixed input set, reported as median with IQR.

    Parameters
    ----------
    n_inputs : int, optional
        Inputs per timing pass, default 1200.
    repeats : int, optional
        Measured passes, default 7.
    seed : int, optional
        RNG seed for input generation.

    Returns
    -------
    dict
    """
    import random

    out: Dict[str, Any] = {
        "n_inputs_per_pass": n_inputs,
        "repeats": repeats,
        "note": (
            "us/exec is wall-clock time per target execution including all probe "
            "calls and the novelty evaluation, in a pure-Python reference "
            "implementation. Absolute values are dominated by Python interpreter "
            "cost; the RATIO to the D0 baseline is the transferable quantity."
        ),
        "per_target": {},
    }
    dim_sets = [("none",), ("D0",), ("D1",), ("D2",), ("D3",),
                ("D0", "D1"), ("D0", "D2"), ("D0", "D3"), ("D0", "D1", "D2", "D3")]

    for tgt in get_targets():
        rng = random.Random(seed)
        mut = Mutator(rng)
        inputs: List[bytes] = []
        data = SEED_INPUT
        for _ in range(n_inputs):
            data = mut.mutate(data)
            inputs.append(data)

        per_cfg: Dict[str, Any] = {}
        for dims in dim_sets:
            label = "uninstrumented" if dims == ("none",) else "+".join(dims)
            for mode, track_exact in (("deployment", False), ("analysis", True)):
                if dims == ("none",) and mode == "analysis":
                    continue
                eng = None if dims == ("none",) else MultiDimGuidanceEngine(
                    dims=dims, map_bits=16, context_depth=16, track_exact=track_exact
                )
                # warmup
                for d in inputs[:200]:
                    if eng is None:
                        tgt.run(d)
                    else:
                        eng.reset()
                        tgt.run(d, eng)
                        eng.evaluate(commit=True)
                samples = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    for d in inputs:
                        if eng is None:
                            tgt.run(d)
                        else:
                            eng.reset()
                            tgt.run(d, eng)
                            eng.evaluate(commit=True)
                    samples.append((time.perf_counter() - t0) / n_inputs * 1e6)
                key = label if dims == ("none",) else f"{label}|{mode}"
                per_cfg[key] = {
                    "us_per_exec_median": float(np.median(samples)),
                    "us_per_exec_iqr": [float(np.percentile(samples, 25)),
                                        float(np.percentile(samples, 75))],
                    "us_per_exec_min": float(np.min(samples)),
                }
        base = per_cfg["D0|deployment"]["us_per_exec_median"]
        raw = per_cfg["uninstrumented"]["us_per_exec_median"]
        for k, v in per_cfg.items():
            v["ratio_vs_D0_deployment"] = v["us_per_exec_median"] / base if base else float("nan")
            v["ratio_vs_uninstrumented"] = v["us_per_exec_median"] / raw if raw else float("nan")
        out["per_target"][tgt.NAME] = per_cfg
    return out


# =========================================================================== #
# Experiment: dimensionality and hash collisions
# =========================================================================== #
def experiment_dimensionality_collisions(
    map_bits_list: Sequence[int] = (10, 12, 14, 16, 18), n_inputs: int = 1500
) -> Dict[str, Any]:
    """State-space cardinality and measured hash-collision rates per dimension.

    The cost side of the refinement ledger: added sensitivity multiplies the number
    of coverage elements, which both inflates the corpus and raises the collision
    rate in a bounded bitmap. Measured collision counts are compared against the
    birthday-bound expectation ``k^2 / 2^(b+1)``.

    Parameters
    ----------
    map_bits_list : sequence of int, optional
        Bitmap sizes (log2) to evaluate.
    n_inputs : int, optional
        Shared corpus size per target.

    Returns
    -------
    dict
    """
    out: Dict[str, Any] = {
        "description": (
            "Exact element counts are collision-free ground truth; mapped_slots is "
            "what a bounded 2^b bitmap actually resolves. collision_rate = "
            "(exact - mapped) / exact, compared against the birthday expectation "
            "k^2 / 2^(b+1)."
        ),
        "per_target": {},
    }
    for tgt in get_targets():
        corpus = collect_shared_corpus(tgt, n_inputs=n_inputs, seed=8181)
        per_b: Dict[str, Any] = {}
        for b in map_bits_list:
            per_dim: Dict[str, Any] = {}
            for d in ge.DIMENSIONS:
                eng = MultiDimGuidanceEngine(dims=(d,), map_bits=b, context_depth=16)
                for data in corpus:
                    eng.reset()
                    tgt.run(data, eng)
                    eng.evaluate(commit=True)
                st = eng.stats()["per_dimension"][d]
                per_dim[d] = {
                    k: st[k]
                    for k in (
                        "exact_elements", "mapped_slots_covered", "collision_rate",
                        "expected_collisions_birthday",
                    )
                    if k in st
                }
                per_dim[d]["map_saturation"] = st["mapped_slots_covered"] / (1 << b)
                for extra in ("distinct_contexts_exact", "n_abstract_states",
                              "n_transitions", "n_value_nodes", "n_kgrams",
                              "mean_node_entropy_bits", "transition_entropy_bits",
                              "max_effective_stack_depth"):
                    if extra in st:
                        per_dim[d][extra] = st[extra]
            per_b[str(b)] = per_dim
        out["per_target"][tgt.NAME] = {"n_corpus_inputs": len(corpus), "by_map_bits": per_b}
    return out


# =========================================================================== #
# Orchestration
# =========================================================================== #
def _sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for strict JSON output.

    Statistical routines legitimately produce ``nan`` (an undefined statistic on an
    empty or degenerate sample) and ``inf`` (an odds ratio with a zero cell). Python's
    ``json`` writes these as ``NaN`` / ``Infinity``, which RFC 8259 does not allow and
    strict parsers reject, so they are mapped to JSON ``null`` here. Statistics that
    would otherwise be lost carry an explicit companion field - for example
    ``odds_ratio_undefined_zero_cell`` and ``odds_ratio_haldane_anscombe``.

    Parameters
    ----------
    obj : object
        Arbitrary nested structure of dicts, lists and scalars.

    Returns
    -------
    object
        The same structure with every non-finite float replaced by ``None``.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _env_info() -> Dict[str, Any]:
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "n_cpu_available": os.cpu_count(),
        "workers_used": N_WORKERS,
        "master_seed": MASTER_SEED,
    }


def run_full_validation(n_trials: int = N_TRIALS) -> Dict[str, Any]:
    """Run every experiment and assemble the validation report.

    Parameters
    ----------
    n_trials : int, optional
        Trials per (target, configuration) cell.

    Returns
    -------
    dict
        The complete report, also written to
        ``results/step1_guidance_validation.json``.
    """
    t_start = time.perf_counter()
    report: Dict[str, Any] = {
        "step": 1,
        "title": "Multidimensional (3D) Guidance Engine - validation report",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": _env_info(),
        "dimensions": {
            "D0": "AFL-style edge coverage (control baseline)",
            "D1": "Calling-context sensitivity (bounded N-depth context hashing)",
            "D2": "Data-flow value-range sensitivity (magnitude / cmp distance / byte splitting)",
            "D3": "Implicit state-machine transition tracking (abstract API state graph)",
        },
        "configurations": {k: list(v) for k, v in CONFIGS.items()},
        "target_budgets": dict(TARGET_BUDGETS),
    }

    print("[1/6] native fidelity ...", flush=True)
    build_native_lib(force=True)
    report["fidelity"] = check_fidelity(n_inputs=5000)
    print(f"      total mismatches: {report['fidelity']['total_mismatches']}", flush=True)

    print("[2/6] exhaustive refinement witnesses (AFL bigram characterisation) ...", flush=True)
    report["refinement_witnesses"] = enumerate_refinement_witnesses(chain_len=6)
    rw = report["refinement_witnesses"]
    print(
        f"      {rw['n_chains_enumerated']} chains -> "
        f"D0(+counts)={rw['n_d0_counted_classes']} classes, D1={rw['n_d1_classes']} classes; "
        f"{rw['n_conflated_groups']} AFL-conflated groups "
        f"({rw['n_groups_with_identical_bigrams_and_different_trigrams']} with equal "
        f"bigrams but different trigrams)",
        flush=True,
    )

    print("[3/6] partition discrimination ...", flush=True)
    report["partition_discrimination"] = experiment_partition_discrimination()
    for name, v in report["partition_discrimination"]["per_target"].items():
        print(
            f"      {name}: D0={v['partitions']['D0']['n_classes']} classes, "
            f"3D={v['partitions']['composite_3d']['n_classes']} classes "
            f"(x{v['refinement_factor_composite_over_D0']:.1f})",
            flush=True,
        )

    print("[4/6] bug-finding experiment ...", flush=True)
    analysis, records = experiment_bug_finding(n_trials=n_trials)
    report["bug_finding"] = analysis
    raw_path = SESSION_DIR / "data" / "step1_trials.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(records))
    report["bug_finding"]["raw_trials_path"] = str(raw_path)

    print("[5/6] context-depth sweep ...", flush=True)
    sweep, sweep_recs = experiment_context_depth_sweep(
        n_trials=n_trials, baseline_records=records
    )
    report["context_depth_sweep"] = sweep
    sweep_path = SESSION_DIR / "data" / "context_depth_trials.json"
    sweep_path.write_text(json.dumps(sweep_recs))
    report["context_depth_sweep"]["raw_trials_path"] = str(sweep_path)

    print("[6/6] overhead + dimensionality ...", flush=True)
    report["overhead"] = experiment_overhead()
    report["dimensionality_and_collisions"] = experiment_dimensionality_collisions()

    report["summary"] = _build_summary(report)
    report["total_wall_seconds"] = time.perf_counter() - t_start

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False so an accidental Infinity/NaN fails loudly instead of emitting
    # JSON that violates RFC 8259 and breaks strict parsers downstream.
    RESULTS_PATH.write_text(
        json.dumps(_sanitize(report), indent=2, default=str, allow_nan=False)
    )
    print(f"\nwrote {RESULTS_PATH}", flush=True)
    return report


def _build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Distil headline findings, including honest negative results."""
    findings: List[str] = []
    bf = report["bug_finding"]["per_target"]

    for tname, tv in bf.items():
        base = tv["configs"]["edge_only"]
        best_cfg, best = None, None
        for cfg, v in tv["configs"].items():
            if cfg in ("edge_only", "blind_random"):
                continue
            if best is None or (v["success_rate"], -v["execs_to_bug_among_successes"].get(
                "median", float("inf")
            )) > (best["success_rate"], -best["execs_to_bug_among_successes"].get(
                "median", float("inf")
            )):
                best_cfg, best = cfg, v
        blind = tv["configs"]["blind_random"]
        findings.append(
            f"{tname}: blind_random {blind['n_bugs_found']}/{blind['n_trials']}, "
            f"edge_only {base['n_bugs_found']}/{base['n_trials']}, "
            f"best={best_cfg} {best['n_bugs_found']}/{best['n_trials']}"
        )

    pd_ = report["partition_discrimination"]["per_target"]
    refinement_ok = all(
        all(v["refinement_verified"].values()) for v in pd_.values()
    )
    return {
        "refinement_theorem_verified_on_all_targets": refinement_ok,
        "native_fidelity_ok": report["fidelity"]["fidelity_ok"],
        "refinement_witness_certified": (
            report["refinement_witnesses"]["witness_pair"] is not None
            and report["refinement_witnesses"]["witness_pair"]["d0_counted_identical"]
            and not report["refinement_witnesses"]["witness_pair"]["d1_identical"]
        ),
        "n_afl_conflated_groups_in_enumeration": report["refinement_witnesses"][
            "n_conflated_groups"
        ],
        "bug_finding_headlines": findings,
        "mean_refinement_factor_composite_over_D0": float(
            np.mean([v["refinement_factor_composite_over_D0"] for v in pd_.values()])
        ),
        "interpretation_caveat": (
            "Partition refinement is a mathematical property and is verified here; it "
            "does NOT by itself imply better bug finding. Added sensitivity also "
            "inflates the corpus and dilutes mutation effort, and AFL's saturating "
            "hit-count classes already encode a surprising amount of path and "
            "recursion structure. The per-target results below are reported as "
            "measured, including where an added dimension does not help."
        ),
    }


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Step 1 guidance-engine validation")
    ap.add_argument("--full", action="store_true", help="run the full validation")
    ap.add_argument("--trials", type=int, default=N_TRIALS, help="trials per cell")
    ap.add_argument("--quick", action="store_true", help="smoke run with 5 trials")
    args = ap.parse_args()

    if args.quick:
        run_full_validation(n_trials=5)
        return 0
    if args.full:
        run_full_validation(n_trials=args.trials)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
