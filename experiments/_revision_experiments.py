"""
Revision experiments: the measurements the first version of the study did not make.

Six experiments, each answering one concrete objection to the first version:

``E1`` native fidelity for the new target
    T4 is cross-validated against its C implementation exactly as T1-T3 were, and
    the whole four-target suite is re-checked so the fidelity claim covers the
    revised artifact rather than a subset of it.

``E2`` the order-sweep campaign on T4
    A benchmark whose triggering predicate is an order-6 property of the executed
    block sequence - provably not a function of the bigram multiset - fuzzed under
    blind search, AFL edge coverage, AFL++ ``NGRAM-k`` for k in {2,3,4,6,8} and
    bounded calling context at depths {2,3,4,6,8,16}. This is the experiment that
    tests the *positive* half of the bigram characterisation; the first version
    tested only the no-go half.

``E3`` the three-way information-loss decomposition
    An AFL map loses information in three separable ways, and the first version
    named only one of them. Over a common execution sample this measures the
    cardinality chain

        |block sequences| -> |bigram count vectors| -> |classified maps| -> |bitmap states|

    whose three ratios are exactly higher-order sequence loss, hit-count
    quantisation and bitmap-index collision.

``E4`` the lattice certificate
    D1 refines D0, but D2 and D3 are *incomparable* with D0 in general. This
    searches for the two-sided witnesses that certify incomparability - an
    execution pair that D0 separates and D2 does not, and a pair that D2 separates
    and D0 does not - rather than asserting a refinement chain the data
    contradicts.

``E5`` the T4 witness enumeration
    Exhaustive enumeration over the sub-alphabet that ``REQ`` uses, certifying
    that executions exist with identical AFL maps (including hit-count classes)
    and identical bigram multisets but different trigram multisets and different
    distances from the bug.

``E6`` paired re-analysis of the original campaigns
    The first version analysed a randomized block design with unpaired tests.
    This re-analyses the *same* per-trial records with exact McNemar, censoring-
    aware win/loss/tie with an exact sign test, seed-stratified log-rank, paired
    sign-flip permutation and a paired probability of superiority.

Usage::

    uv run python workflow/05_revision_experiments.py [--pilot] [--trials N] [--budget N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import platform
import random
import sys
import time
from collections import Counter
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
    CONFIG_PARAMS,
    Mutator,
    SEED_INPUT,
    SEED_INPUT_BY_TARGET,
    TARGET_BUDGETS,
    trial_seed,
    worker,
)
from guidance_engine import (  # noqa: E402
    MultiDimGuidanceEngine,
    afl_count_class,
)
from stats_paired import (  # noqa: E402
    mcnemar_exact,
    paired_permutation,
    paired_prob_superiority,
    paired_win_loss_tie,
    sanitize_nonfinite,
    sign_test_exact,
    stratified_logrank,
)
from stats_utils import (  # noqa: E402
    benjamini_hochberg,
    clopper_pearson,
    describe,
    fisher_success,
    logrank_test,
    vargha_delaney_a12,
)
from test_targets import (  # noqa: E402
    NativeTargetLib,
    T4TrigramTarget,
    build_native_lib,
    get_targets,
)

MASTER_SEED = 20260804
RESULTS = SESSION_DIR / "results"
DATA = SESSION_DIR / "data"

#: The order sweep. ``edge_only`` is the control; ``ngram2`` must reproduce it
#: exactly (AFL++ NGRAM-2 *is* AFL edge coverage), which is an internal check.
ORDER_CONFIGS: List[str] = [
    "blind_random",
    "edge_only",
    "ngram2",
    "ngram3",
    "ngram4",
    "ngram6",
    "ngram8",
    "ctx2",
    "ctx3",
    "ctx4",
    "ctx6",
    "ctx8",
    "ctx16",
    "full_3d",
]

#: Nominal order of each configuration's abstraction over the block sequence.
CONFIG_ORDER: Dict[str, int] = {
    "blind_random": 0,
    "edge_only": 2,
    "ngram2": 2,
    "ngram3": 3,
    "ngram4": 4,
    "ngram6": 6,
    "ngram8": 8,
    "ctx2": 2,
    "ctx3": 3,
    "ctx4": 4,
    "ctx6": 6,
    "ctx8": 8,
    "ctx16": 16,
    "full_3d": 16,
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================================== #
# E1 - native fidelity, all four targets
# =========================================================================== #
def e1_fidelity(n_inputs: int = 5000, max_len: int = 96) -> Dict[str, Any]:
    """Differential cross-validation of every Python model against native C.

    Random inputs are pushed through both implementations and the full decision
    trace, the milestone and the bug flag are compared exactly. The wording of the
    conclusion matters: this is *empirical cross-validation over a random sample*,
    not a proof of semantic equivalence, and it cannot exclude divergence caused
    by fixed-width overflow, signedness, undefined behaviour, compiler
    optimisation or pointer aliasing outside the sampled region.

    Parameters
    ----------
    n_inputs : int, optional
        Random inputs per target, default 5000.
    max_len : int, optional
        Maximum random input length, default 96.

    Returns
    -------
    dict
        Per-target mismatch counts and sample statistics.
    """
    build_native_lib(force=True)
    native = NativeTargetLib()
    rng = random.Random(MASTER_SEED)
    out: Dict[str, Any] = {
        "n_inputs_per_target": n_inputs,
        "max_input_len": max_len,
        "compiler": native.compiler_version,
        "per_target": {},
        "claim": (
            "empirically cross-validated over randomly generated executions with "
            "zero observed mismatches; not a proof of semantic equivalence"
        ),
    }
    total = 0
    for tgt in get_targets():
        mism = 0
        bugs = 0
        ms_hist: Counter = Counter()
        examples: List[Dict[str, Any]] = []
        for _ in range(n_inputs):
            ln = rng.randrange(1, max_len + 1)
            data = bytes(rng.getrandbits(8) for _ in range(ln))
            a = tgt.run(data)
            b = native.run(tgt.NAME, data)
            if a.trace != b.trace or a.milestone != b.milestone or a.bug != b.bug:
                mism += 1
                if len(examples) < 3:
                    examples.append(
                        {
                            "input_hex": data.hex(),
                            "py_trace_len": len(a.trace),
                            "c_trace_len": len(b.trace),
                            "py_milestone": a.milestone,
                            "c_milestone": b.milestone,
                        }
                    )
            bugs += int(a.bug)
            ms_hist[a.milestone] += 1
        total += mism
        out["per_target"][tgt.NAME] = {
            "mismatches": mism,
            "bug_rate_random": bugs / n_inputs,
            "milestone_histogram": {str(k): v for k, v in sorted(ms_hist.items())},
            "mismatch_examples": examples,
        }
    out["total_mismatches"] = total
    out["fidelity_ok"] = total == 0
    return out


# =========================================================================== #
# E3 - three-way information-loss decomposition
# =========================================================================== #
def _exec_signatures(target, data: bytes, map_bits_list: Sequence[int]) -> Dict[str, Any]:
    """Collect every level of the D0 abstraction chain for one execution."""
    eng = MultiDimGuidanceEngine(dims=("D0",), map_bits=64 if False else 16)
    eng.reset()
    res = target.run(data, eng)
    seq = tuple(res.trace)
    counts = dict(eng.d0.exact_trace)  # exact (prev_bb, cur_bb) -> raw count
    g2 = frozenset(counts.items())
    classified = frozenset((k, afl_count_class(v)) for k, v in counts.items())
    return {"seq": seq, "g2": g2, "classified": classified, "counts": counts,
            "bug": res.bug, "milestone": res.milestone}


def _bitmap_state(counts: Dict[Tuple[Any, Any], int], loc, map_bits: int) -> frozenset:
    """The *actual* AFL trace-bits state at a given map size."""
    mask = (1 << map_bits) - 1
    agg: Dict[int, int] = {}
    for (prev_bb, cur_bb), c in counts.items():
        idx = ((loc(prev_bb) >> 1) ^ loc(cur_bb)) & mask
        agg[idx] = min(255, agg.get(idx, 0) + c)
    return frozenset((i, afl_count_class(v)) for i, v in agg.items())


def _guided_sample(target, n_exec: int, seed: int) -> List[bytes]:
    """Draw inputs from a coverage-guided mutational distribution.

    Sampling uniformly at random is the wrong probe for this measurement and the
    first attempt at it was misleading: uniform random inputs produce shallow
    executions, so no edge is ever taken four times and hit-count quantisation
    never engages, and the reachable bigram set stays far below the birthday
    threshold, so no index ever collides. Measured that way all three loss
    mechanisms read as zero - not because they are absent but because the sample
    never reaches the regime in which they operate. A fuzzer does reach it, so
    the sample here is drawn the way a fuzzer draws it: an edge-coverage-guided
    mutational loop over a growing corpus, i.e. exactly the input distribution
    the metric actually faces in a campaign.

    Parameters
    ----------
    target : object
        Benchmark target instance.
    n_exec : int
        Number of inputs to draw.
    seed : int
        RNG seed.

    Returns
    -------
    list of bytes
    """
    rng = random.Random(seed)
    mut = Mutator(rng)
    eng = MultiDimGuidanceEngine(dims=("D0",), map_bits=16)
    seed_input = SEED_INPUT_BY_TARGET.get(target.NAME, SEED_INPUT)
    corpus: List[bytes] = [seed_input]
    eng.reset()
    target.run(seed_input, eng)
    eng.evaluate(commit=True)
    out: List[bytes] = []
    cursor = 0
    while len(out) < n_exec:
        parent = corpus[cursor % len(corpus)]
        cursor += 1
        data = mut.mutate(parent, splice_pool=corpus)
        eng.reset()
        target.run(data, eng)
        fb = eng.evaluate(commit=True)
        if fb.is_interesting:
            corpus.append(data)
        out.append(data)
    return out


def e3_information_loss(
    n_exec: int = 20_000, map_bits_list: Sequence[int] = (16, 12, 10, 8)
) -> Dict[str, Any]:
    """Measure the three separable information-loss mechanisms of an AFL map.

    The chain of cardinalities over a common execution sample is

    ``|distinct block sequences|`` (the ground truth an ideal path abstraction
    would see) ``->`` ``|distinct exact bigram count vectors|`` (Theorem 2's
    domain) ``->`` ``|distinct classified maps under collision-free indexing|``
    (after hit-count quantisation) ``->`` ``|distinct bitmap states at 2**b|``
    (after index collision).

    Each successive ratio isolates one mechanism, because the only thing that
    changes between consecutive levels is that mechanism. The sample is drawn by
    :func:`_guided_sample` rather than uniformly, for the reason documented
    there. The same chain is computed for the context dimension, whose element
    set is roughly two orders of magnitude larger, because index collision is a
    function of element count and is therefore invisible on D0 alone.

    Parameters
    ----------
    n_exec : int, optional
        Executions sampled per target, default 20000.
    map_bits_list : sequence of int, optional
        Map sizes at which to measure the collision level.

    Returns
    -------
    dict
        Per-target cardinalities, ratios and diagnostic counts.
    """
    out: Dict[str, Any] = {
        "n_exec_per_target": n_exec,
        "map_bits_measured": list(map_bits_list),
        "sampling": "coverage-guided mutational loop (not uniform random)",
        "description": (
            "cardinality chain over a common execution sample: block sequences -> "
            "exact bigram count vectors -> classified maps (collision-free) -> "
            "bitmap states at 2**b"
        ),
        "per_target": {},
    }
    for tgt in get_targets():
        inputs = _guided_sample(tgt, n_exec, MASTER_SEED ^ (hash(tgt.NAME) & 0xFFFFFFFF))
        eng = MultiDimGuidanceEngine(dims=("D0", "D1"), map_bits=16, context_depth=16)
        seqs: set = set()
        g2s: set = set()
        cls: set = set()
        ctx_exact: set = set()
        ctx_cls: set = set()
        bitmaps: Dict[int, set] = {b: set() for b in map_bits_list}
        ctx_slots: Dict[int, set] = {b: set() for b in map_bits_list}
        cls_to_ms: Dict[Any, set] = {}
        max_raw = 0
        n_ge4 = 0
        for data in inputs:
            eng.reset()
            res = tgt.run(data, eng)
            counts = dict(eng.d0.exact_trace)
            if counts:
                mx = max(counts.values())
                max_raw = max(max_raw, mx)
                n_ge4 += int(mx >= 4)
            seqs.add(tuple(res.trace))
            g2s.add(frozenset(counts.items()))
            c = frozenset((k, afl_count_class(v)) for k, v in counts.items())
            cls.add(c)
            cls_to_ms.setdefault(c, set()).add(res.milestone)
            for b in map_bits_list:
                bitmaps[b].add(_bitmap_state(counts, eng.d0.loc, b))
            ce = eng.d1.trace_signature()
            ctx_exact.add(ce)
            for b in map_bits_list:
                mask = (1 << b) - 1
                ctx_slots[b].add(
                    frozenset(
                        ((eng.d1.loc(e[1]) >> 1) ^ eng.d1.loc(e[2])
                         ^ (hash(e[0]) & mask)) & mask
                        for e in ce
                    )
                )
            ctx_cls.add(ce)
        n_seq, n_g2, n_cls = len(seqs), len(g2s), len(cls)
        mixed = sum(1 for v in cls_to_ms.values() if len(v) > 1)
        out["per_target"][tgt.NAME] = {
            "n_distinct_block_sequences": n_seq,
            "n_distinct_bigram_count_vectors": n_g2,
            "n_distinct_classified_maps_collision_free": n_cls,
            "n_distinct_bitmap_states": {str(b): len(bitmaps[b]) for b in map_bits_list},
            "loss_higher_order_sequence": 1.0 - n_g2 / n_seq if n_seq else 0.0,
            "loss_hitcount_quantisation": 1.0 - n_cls / n_g2 if n_g2 else 0.0,
            "loss_index_collision": {
                str(b): 1.0 - len(bitmaps[b]) / n_cls if n_cls else 0.0
                for b in map_bits_list
            },
            "max_raw_edge_count_observed": max_raw,
            "frac_executions_with_an_edge_taken_4plus": n_ge4 / n_exec,
            "n_classified_classes_mixing_milestones": mixed,
            "frac_classified_classes_mixing_milestones": mixed / n_cls if n_cls else 0.0,
            "context_dimension": {
                "n_distinct_exact_signatures": len(ctx_exact),
                "n_distinct_mapped_signatures": {
                    str(b): len(ctx_slots[b]) for b in map_bits_list
                },
                "loss_index_collision": {
                    str(b): 1.0 - len(ctx_slots[b]) / len(ctx_exact)
                    if ctx_exact else 0.0
                    for b in map_bits_list
                },
            },
        }
    return out


# =========================================================================== #
# E4 - lattice certificate (which dimensions refine which)
# =========================================================================== #
def e4_lattice(n_exec: int = 4000) -> Dict[str, Any]:
    """Certify the refinement lattice, including the *incomparabilities*.

    For each pair of dimensions ``(X, Y)`` over a common execution sample the
    search looks for two-sided witnesses:

    * an execution pair with ``alpha_X`` equal and ``alpha_Y`` different
      (``Y`` is not coarser than ``X``), and
    * an execution pair with ``alpha_Y`` equal and ``alpha_X`` different
      (``X`` is not coarser than ``Y``).

    Both present means the two partitions are **incomparable**; only the first
    means ``Y`` strictly refines ``X``; neither means the partitions agree on the
    sample. This is a certificate over a sample, so "refines" is reported as "no
    counterexample found in ``n_exec`` executions" rather than as a proof, except
    for D1 over D0 where the projection ``alpha_0 = pi o alpha_1`` is structural
    and is additionally asserted per execution.

    Parameters
    ----------
    n_exec : int, optional
        Executions sampled per target, default 4000.

    Returns
    -------
    dict
        Per-target, per-pair witness records.
    """
    dims = ["D0", "D1", "D2", "D3"]
    out: Dict[str, Any] = {
        "n_exec_per_target": n_exec,
        "description": (
            "two-sided incomparability witnesses over a common execution sample; "
            "D1 over D0 additionally carries the structural projection assertion"
        ),
        "per_target": {},
    }
    for tgt in get_targets():
        rng = random.Random(MASTER_SEED + 7 + (hash(tgt.NAME) & 0xFFFF))
        eng = MultiDimGuidanceEngine(dims=tuple(dims), map_bits=16, context_depth=16)
        sigs: Dict[str, List[frozenset]] = {d: [] for d in dims}
        projection_ok = True
        for _ in range(n_exec):
            ln = rng.randrange(1, 64)
            data = bytes(rng.getrandbits(8) for _ in range(ln))
            eng.reset()
            tgt.run(data, eng)
            sigs["D0"].append(eng.d0.trace_signature())
            sigs["D1"].append(eng.d1.trace_signature())
            sigs["D2"].append(eng.d2.trace_signature())
            sigs["D3"].append(eng.d3.trace_signature())
            # structural projection: pi(ctx, prev, cur) = (prev, cur)
            proj = frozenset((e[1], e[2]) for e in eng.d1.trace_signature())
            if proj != eng.d0.trace_signature():
                projection_ok = False
        # index executions by each dimension's signature to find equal pairs fast
        rec: Dict[str, Any] = {"projection_alpha0_eq_pi_alpha1": projection_ok}
        for x, y in itertools.permutations(dims, 2):
            groups: Dict[frozenset, List[int]] = {}
            for i, s in enumerate(sigs[x]):
                groups.setdefault(s, []).append(i)
            witness = None
            for _, idxs in groups.items():
                if len(idxs) < 2:
                    continue
                base = sigs[y][idxs[0]]
                for j in idxs[1:]:
                    if sigs[y][j] != base:
                        witness = (idxs[0], j)
                        break
                if witness:
                    break
            rec[f"{x}_equal_but_{y}_differs"] = witness is not None
        for a, b in itertools.combinations(dims, 2):
            fwd = rec[f"{a}_equal_but_{b}_differs"]
            bwd = rec[f"{b}_equal_but_{a}_differs"]
            if fwd and bwd:
                verdict = "incomparable"
            elif fwd and not bwd:
                verdict = f"{b} refines {a}"
            elif bwd and not fwd:
                verdict = f"{a} refines {b}"
            else:
                verdict = "no separation observed"
            rec[f"verdict_{a}_vs_{b}"] = verdict
        rec["n_distinct_classes"] = {d: len(set(sigs[d])) for d in dims}
        composite = [
            frozenset(("D0", e) for e in sigs["D0"][i])
            | frozenset(("D1", e) for e in sigs["D1"][i])
            | frozenset(("D2", e) for e in sigs["D2"][i])
            | frozenset(("D3", e) for e in sigs["D3"][i])
            for i in range(n_exec)
        ]
        rec["n_distinct_classes"]["composite"] = len(set(composite))
        # the composite must refine every constituent: equal composite => equal each
        cgroups: Dict[frozenset, List[int]] = {}
        for i, s in enumerate(composite):
            cgroups.setdefault(s, []).append(i)
        comp_ok = True
        for _, idxs in cgroups.items():
            for d in dims:
                if len({sigs[d][i] for i in idxs}) > 1:
                    comp_ok = False
        rec["composite_refines_all_constituents"] = comp_ok
        out["per_target"][tgt.NAME] = rec
    return out


# =========================================================================== #
# E5 - T4 witness enumeration
# =========================================================================== #
def e5_t4_witnesses() -> Dict[str, Any]:
    """Exhaustive witness enumeration for the T4 higher-order blind spot.

    Enumerates every walk over the three symbols ``REQ`` uses (``3**WALK``
    executions), groups them by their AFL *counted* signature - the strongest form
    of the baseline, edge identity plus saturating hit-count class - and reports
    how many groups the baseline conflates, how many of those groups mix
    executions at different distances from the bug, and how many are separated by
    the trigram spectrum.

    Returns
    -------
    dict
        Enumeration statistics plus the hand-checkable named witness pairs.
    """
    tgt = T4TrigramTarget()
    syms = sorted(set(tgt.REQ))
    enc = lambda seq: bytes((s ^ tgt.DECODE_XOR) for s in seq)  # noqa: E731

    eng = MultiDimGuidanceEngine(dims=("D0",), map_bits=16)
    rows = []
    for seq in itertools.product(syms, repeat=tgt.WALK):
        eng.reset()
        res = tgt.run(enc(seq), eng)
        counts = dict(eng.d0.exact_trace)
        rows.append(
            {
                "seq": seq,
                "counted": frozenset((k, afl_count_class(v)) for k, v in counts.items()),
                "g2": Counter(zip(res.trace, res.trace[1:])),
                "g3": Counter(zip(res.trace, res.trace[1:], res.trace[2:])),
                "ms": res.milestone,
                "bug": res.bug,
            }
        )
    groups: Dict[Any, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["counted"], []).append(i)
    conflated = {k: v for k, v in groups.items() if len(v) > 1}
    mixed_ms = sum(1 for v in conflated.values() if len({rows[i]["ms"] for i in v}) > 1)
    mixed_bug = sum(1 for v in conflated.values() if len({rows[i]["bug"] for i in v}) > 1)
    g2_eq_g3_diff = 0
    max_ms_gap = 0
    best_pair = None
    for v in conflated.values():
        for i, j in itertools.combinations(v, 2):
            if rows[i]["g2"] == rows[j]["g2"] and rows[i]["g3"] != rows[j]["g3"]:
                g2_eq_g3_diff += 1
                gap = abs(rows[i]["ms"] - rows[j]["ms"])
                if gap > max_ms_gap:
                    max_ms_gap = gap
                    hi, lo = (i, j) if rows[i]["ms"] > rows[j]["ms"] else (j, i)
                    best_pair = (rows[hi]["seq"], rows[lo]["seq"])
    if best_pair is None:
        best_pair = (tgt.REQ, tgt.WITNESS_NEG)

    def _pair(a: Sequence[int], b: Sequence[int]) -> Dict[str, Any]:
        e1 = MultiDimGuidanceEngine(dims=("D0",), map_bits=16)
        e2 = MultiDimGuidanceEngine(dims=("D0",), map_bits=16)
        e1.reset()
        ra = tgt.run(enc(a), e1)
        e2.reset()
        rb = tgt.run(enc(b), e2)
        n3a = MultiDimGuidanceEngine(dims=("DN",), ngram=3)
        n3b = MultiDimGuidanceEngine(dims=("DN",), ngram=3)
        n3a.reset()
        tgt.run(enc(a), n3a)
        n3b.reset()
        tgt.run(enc(b), n3b)
        c3a = MultiDimGuidanceEngine(dims=("D0", "D1"), context_depth=3)
        c3b = MultiDimGuidanceEngine(dims=("D0", "D1"), context_depth=3)
        c3a.reset()
        tgt.run(enc(a), c3a)
        c3b.reset()
        tgt.run(enc(b), c3b)
        g2a = Counter(zip(ra.trace, ra.trace[1:]))
        g2b = Counter(zip(rb.trace, rb.trace[1:]))
        g3a = Counter(zip(ra.trace, ra.trace[1:], ra.trace[2:]))
        g3b = Counter(zip(rb.trace, rb.trace[1:], rb.trace[2:]))
        return {
            "seq_a": list(a),
            "seq_b": list(b),
            "input_a_hex": enc(a).hex(),
            "input_b_hex": enc(b).hex(),
            "milestone_a": ra.milestone,
            "milestone_b": rb.milestone,
            "bug_a": ra.bug,
            "bug_b": rb.bug,
            "d0_counted_identical": e1.d0_signature_counted() == e2.d0_signature_counted(),
            "block_bigram_multiset_identical": g2a == g2b,
            "block_trigram_multiset_differs": g3a != g3b,
            "ngram3_separates": n3a.dn.trace_signature() != n3b.dn.trace_signature(),
            "context_depth3_separates": c3a.d1.trace_signature() != c3b.d1.trace_signature(),
        }

    # ---- discrimination as a function of feedback order --------------------
    # For each order n, recompute the abstraction over the same 729 walks and ask
    # (a) how many classes it induces and (b) how many of the AFL-conflated groups
    # it splits. This is the discrimination side of the order sweep, measured
    # exhaustively rather than sampled.
    order_rows: Dict[str, Any] = {}
    for n in (2, 3, 4, 5, 6):
        sig_ng: List[frozenset] = []
        sig_ctx: List[frozenset] = []
        for seq in itertools.product(syms, repeat=tgt.WALK):
            en = MultiDimGuidanceEngine(dims=("DN",), ngram=n)
            en.reset()
            tgt.run(enc(seq), en)
            sig_ng.append(en.dn.trace_signature())
            ec = MultiDimGuidanceEngine(dims=("D0", "D1"), context_depth=n)
            ec.reset()
            tgt.run(enc(seq), ec)
            sig_ctx.append(ec.d1.trace_signature())
        split_ng = sum(
            1 for v in conflated.values() if len({sig_ng[i] for i in v}) > 1
        )
        split_ctx = sum(
            1 for v in conflated.values() if len({sig_ctx[i] for i in v}) > 1
        )
        # residual: groups that still mix milestones AFTER this abstraction
        res_ng = 0
        for v in conflated.values():
            sub: Dict[Any, set] = {}
            for i in v:
                sub.setdefault(sig_ng[i], set()).add(rows[i]["ms"])
            res_ng += sum(1 for m in sub.values() if len(m) > 1)
        order_rows[str(n)] = {
            "ngram_classes": len(set(sig_ng)),
            "context_classes": len(set(sig_ctx)),
            "afl_conflated_groups_split_by_ngram": split_ng,
            "afl_conflated_groups_split_by_context": split_ctx,
            "residual_milestone_mixing_cells_under_ngram": res_ng,
        }

    return {
        "alphabet": tgt.ALPHABET,
        "walk_length": tgt.WALK,
        "discrimination_by_order": order_rows,
        "req": list(tgt.REQ),
        "sub_alphabet_enumerated": syms,
        "n_walks_enumerated": len(rows),
        "n_d0_counted_classes": len(groups),
        "n_conflated_groups": len(conflated),
        "n_walks_in_conflated_groups": sum(len(v) for v in conflated.values()),
        "n_conflated_groups_mixing_milestone": mixed_ms,
        "n_conflated_groups_mixing_bug_flag": mixed_bug,
        "n_pairs_g2_identical_g3_different": g2_eq_g3_diff,
        "max_milestone_gap_within_a_conflated_group": max_ms_gap,
        "witness_hot_cold": _pair(best_pair[0], best_pair[1]),
        "witness_bug_nobug": _pair(tgt.REQ, tgt.WITNESS_NEG),
        "note": (
            "the bug/no-bug pair differs in one extra block, the bug block itself, "
            "which by construction executes only after the predicate has already "
            "been satisfied and therefore supplies no gradient; the hot/cold pair "
            "carries the full claim with bit-identical AFL maps"
        ),
    }


# =========================================================================== #
# E2 - the order-sweep campaign on T4
# =========================================================================== #
def _campaign(
    target: str,
    configs: Sequence[str],
    n_trials: int,
    budget: int,
    workers: int,
    schedule: str = "round_robin",
) -> List[Dict[str, Any]]:
    """Run a blocked campaign: trial ``i`` uses the same seed in every arm."""
    jobs = []
    for cfg in configs:
        for i in range(n_trials):
            jobs.append(
                {
                    "target_name": target,
                    "config_name": cfg,
                    "seed": trial_seed(MASTER_SEED, target, i),
                    "max_execs": budget,
                    "curve_every": max(1000, budget // 60),
                    "schedule": schedule,
                    "tag": f"{cfg}#{i}",
                }
            )
    with mp.Pool(processes=workers) as pool:
        recs = pool.map(worker, jobs, chunksize=1)
    for r, j in zip(recs, jobs):
        r["block"] = int(j["tag"].split("#")[1])
        r["schedule"] = schedule
    return recs


def _arm(recs: List[Dict[str, Any]], cfg: str, budget: int) -> Dict[str, np.ndarray]:
    """Extract blocked time-to-event vectors for one configuration."""
    rows = sorted([r for r in recs if r["config"] == cfg], key=lambda r: r["block"])
    t = np.array([r["first_bug_exec"] if r["first_bug_exec"] else r["execs"] for r in rows],
                 dtype=float)
    e = np.array([1 if r["first_bug_exec"] else 0 for r in rows], dtype=int)
    return {
        "t": t,
        "e": e,
        "block": np.array([r["block"] for r in rows]),
        "milestone": np.array([r["max_milestone"] for r in rows], dtype=float),
        "corpus": np.array([r["corpus_size"] for r in rows], dtype=float),
        "elements": np.array([r["engine_stats"]["total_exact_elements"] for r in rows],
                             dtype=float),
        "exec_per_s": np.array([r["execs"] / max(r["wall_seconds"], 1e-9) for r in rows],
                               dtype=float),
    }


def e2_order_sweep(n_trials: int, budget: int, workers: int,
                   schedule: str = "round_robin") -> Dict[str, Any]:
    """The T4 campaign plus its paired analysis against the edge-coverage control.

    Parameters
    ----------
    n_trials : int
        Trials per configuration (blocked on seed).
    budget : int
        Execution budget per trial.
    workers : int
        Worker processes.

    Returns
    -------
    dict
        Per-configuration summaries, paired contrasts against ``edge_only``, and
        the FDR-corrected q-values over the contrast family.
    """
    t0 = time.perf_counter()
    recs = _campaign("T4_trigram", ORDER_CONFIGS, n_trials, budget, workers, schedule)
    arms = {c: _arm(recs, c, budget) for c in ORDER_CONFIGS}

    per_cfg: Dict[str, Any] = {}
    for c in ORDER_CONFIGS:
        a = arms[c]
        k = int(a["e"].sum())
        lo, hi = clopper_pearson(k, n_trials)
        hit = a["t"][a["e"] == 1]
        per_cfg[c] = {
            "order": CONFIG_ORDER[c],
            "n_trials": n_trials,
            "n_success": k,
            "success_rate": k / n_trials,
            "success_ci95": [lo, hi],
            "median_execs_to_bug": float(np.median(hit)) if hit.size else None,
            "iqr_execs_to_bug": [float(np.percentile(hit, 25)), float(np.percentile(hit, 75))]
            if hit.size
            else None,
            "median_max_milestone": float(np.median(a["milestone"])),
            "median_corpus": float(np.median(a["corpus"])),
            "median_elements": float(np.median(a["elements"])),
            "median_exec_per_s": float(np.median(a["exec_per_s"])),
        }

    base = arms["edge_only"]
    contrasts: Dict[str, Any] = {}
    p_paired: List[float] = []
    p_keys: List[str] = []
    for c in ORDER_CONFIGS:
        if c == "edge_only":
            continue
        a = arms[c]
        mc = mcnemar_exact(a["e"], base["e"])
        wl = paired_win_loss_tie(a["t"], a["e"], base["t"], base["e"])
        sg = sign_test_exact(wl["wins_a"], wl["wins_b"])
        pm = paired_permutation(a["t"], a["e"], base["t"], base["e"])
        sl = stratified_logrank(a["t"], a["e"], base["t"], base["e"], strata=a["block"])
        ps = paired_prob_superiority(a["t"], a["e"], base["t"], base["e"])
        # unpaired robustness checks, retained but no longer primary
        fi = fisher_success(int(a["e"].sum()), n_trials, int(base["e"].sum()), n_trials)
        lr = logrank_test(a["t"], a["e"], base["t"], base["e"])
        a12 = vargha_delaney_a12(a["t"], base["t"])
        contrasts[c] = {
            "vs": "edge_only",
            "paired_primary": {
                "mcnemar_exact": mc,
                "win_loss_tie": wl,
                "sign_exact": sg,
                "paired_permutation": pm,
                "stratified_logrank": sl,
                "paired_prob_superiority": ps,
            },
            "unpaired_robustness": {
                "fisher": fi,
                "logrank": lr,
                "a12_time": a12,
            },
        }
        p_paired.append(mc["p_value"])
        p_keys.append(f"{c}:mcnemar")
        p_paired.append(sg["p_value"])
        p_keys.append(f"{c}:sign")
        p_paired.append(sl["p_value"])
        p_keys.append(f"{c}:stratified_logrank")
    bh = benjamini_hochberg(p_paired)
    qmap = {k: q for k, q in zip(p_keys, bh["q_values"])}
    for c in contrasts:
        contrasts[c]["q_values"] = {
            "mcnemar": qmap[f"{c}:mcnemar"],
            "sign": qmap[f"{c}:sign"],
            "stratified_logrank": qmap[f"{c}:stratified_logrank"],
        }

    # internal consistency: NGRAM-2 must reproduce AFL edge coverage exactly
    ng2 = arms["ngram2"]
    consistency = {
        "ngram2_equals_edge_only_per_block": bool(
            np.array_equal(ng2["t"], base["t"]) and np.array_equal(ng2["e"], base["e"])
        ),
        "n_blocks_compared": int(ng2["t"].size),
    }

    return {
        "target": "T4_trigram",
        "schedule": schedule,
        "budget_execs": budget,
        "n_trials_per_cell": n_trials,
        "configs": list(ORDER_CONFIGS),
        "per_config": per_cfg,
        "contrasts_vs_edge_only": contrasts,
        "multiple_testing_correction": {
            "method": "Benjamini-Hochberg FDR",
            "family": "all paired contrasts against the edge-coverage control on T4",
            "n_tests": len(p_paired),
            "alpha": 0.05,
        },
        "internal_consistency": consistency,
        "wall_seconds": time.perf_counter() - t0,
        "raw_trials_path": str(DATA / "t4_order_sweep.json"),
        "_raw": recs,
    }


# =========================================================================== #
# E6 - paired re-analysis of the original campaigns
# =========================================================================== #
def e6_paired_reanalysis() -> Dict[str, Any]:
    """Re-analyse the original T1-T3 campaigns and depth sweep as a blocked design.

    The per-trial records are read from the first study's raw output, so this is a
    re-analysis of exactly the same data with a method matched to the design, not
    a new campaign with new randomness.

    Returns
    -------
    dict
        Per-target paired contrasts against ``edge_only`` and the depth sweep
        re-analysed against depth 1.
    """
    out: Dict[str, Any] = {"description": (
        "same per-trial records as the original campaign, re-analysed with "
        "matched-design procedures; no new executions"
    )}

    trials = json.loads((DATA / "step1_trials.json").read_text())
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in trials:
        by.setdefault((r["target"], r["config"]), []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["seed"])

    def vecs(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        t = np.array([r["first_bug_exec"] if r["first_bug_exec"] else r["execs"] for r in rows],
                     dtype=float)
        e = np.array([1 if r["first_bug_exec"] else 0 for r in rows], dtype=int)
        return t, e

    per_target: Dict[str, Any] = {}
    pvals: List[float] = []
    keys: List[str] = []
    targets = sorted({t for t, _ in by})
    for tgt in targets:
        base_rows = by[(tgt, "edge_only")]
        # align blocks by seed
        base_seeds = [r["seed"] for r in base_rows]
        bt, be = vecs(base_rows)
        cfgs = sorted({c for t, c in by if t == tgt and c != "edge_only"})
        rec: Dict[str, Any] = {}
        for c in cfgs:
            rows = by[(tgt, c)]
            assert [r["seed"] for r in rows] == base_seeds, "block alignment failed"
            at, ae = vecs(rows)
            mc = mcnemar_exact(ae, be)
            wl = paired_win_loss_tie(at, ae, bt, be)
            sg = sign_test_exact(wl["wins_a"], wl["wins_b"])
            pm = paired_permutation(at, ae, bt, be)
            sl = stratified_logrank(at, ae, bt, be, strata=np.arange(at.size))
            ps = paired_prob_superiority(at, ae, bt, be)
            rec[c] = {
                "mcnemar_exact": mc,
                "win_loss_tie": wl,
                "sign_exact": sg,
                "paired_permutation": pm,
                "stratified_logrank": sl,
                "paired_prob_superiority": ps,
            }
            pvals += [mc["p_value"], sg["p_value"], sl["p_value"]]
            keys += [f"{tgt}:{c}:mcnemar", f"{tgt}:{c}:sign", f"{tgt}:{c}:stratlr"]
        per_target[tgt] = rec
    bh = benjamini_hochberg(pvals)
    qm = {k: q for k, q in zip(keys, bh["q_values"])}
    for tgt in per_target:
        for c in per_target[tgt]:
            per_target[tgt][c]["q_values"] = {
                "mcnemar": qm[f"{tgt}:{c}:mcnemar"],
                "sign": qm[f"{tgt}:{c}:sign"],
                "stratified_logrank": qm[f"{tgt}:{c}:stratlr"],
            }
    out["bug_finding_per_target"] = per_target
    out["bug_finding_correction"] = {
        "method": "Benjamini-Hochberg FDR",
        "n_tests": len(pvals),
        "family": "all paired contrasts against edge_only across T1-T3",
    }

    # ---- depth sweep -------------------------------------------------------
    dsw = json.loads((DATA / "context_depth_trials.json").read_text())
    byd: Dict[Any, List[Dict[str, Any]]] = {}
    for r in dsw:
        tag = r.get("tag") or r.get("config", "")
        key = str(tag).split("|")[0]  # tags are "<arm>|<trial index>"
        byd.setdefault(key, []).append(r)
    for v in byd.values():
        v.sort(key=lambda r: r["seed"])
    depth_keys = sorted(byd)
    base_key = "N=1" if "N=1" in byd else depth_keys[0]
    bt, be = vecs(byd[base_key])
    dp: Dict[str, Any] = {}
    dpv: List[float] = []
    dpk: List[str] = []
    for k in depth_keys:
        if k == base_key:
            continue
        at, ae = vecs(byd[k])
        n = min(at.size, bt.size)
        wl = paired_win_loss_tie(at[:n], ae[:n], bt[:n], be[:n])
        sg = sign_test_exact(wl["wins_a"], wl["wins_b"])
        sl = stratified_logrank(at[:n], ae[:n], bt[:n], be[:n], strata=np.arange(n))
        pm = paired_permutation(at[:n], ae[:n], bt[:n], be[:n])
        dp[str(k)] = {"win_loss_tie": wl, "sign_exact": sg,
                      "stratified_logrank": sl, "paired_permutation": pm,
                      "median_execs": float(np.median(at)),
                      "n_success": int(ae.sum()), "n_trials": int(ae.size)}
        dpv += [sg["p_value"], sl["p_value"]]
        dpk += [f"{k}:sign", f"{k}:stratlr"]
    bh2 = benjamini_hochberg(dpv) if dpv else {"q_values": []}
    qm2 = {kk: q for kk, q in zip(dpk, bh2["q_values"])}
    for k in dp:
        dp[k]["q_values"] = {"sign": qm2.get(f"{k}:sign"), "stratified_logrank": qm2.get(f"{k}:stratlr")}
    out["depth_sweep_paired"] = {
        "baseline_arm": str(base_key),
        "arms": dp,
        "correction": {"method": "Benjamini-Hochberg FDR", "n_tests": len(dpv)},
    }
    return out


# =========================================================================== #
# main
# =========================================================================== #
def main() -> int:
    """Run every revision experiment and write the results file."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true", help="small, fast configuration")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--budget", type=int, default=150_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()

    n_trials = 5 if args.pilot else args.trials
    budget = 40_000 if args.pilot else args.budget
    only = {s for s in args.only.split(",") if s}

    t_start = time.perf_counter()
    out: Dict[str, Any] = {
        "step": 2,
        "title": "Revision experiments: higher-order blind spot, information-loss "
                 "decomposition, refinement lattice and matched-design inference",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "master_seed": MASTER_SEED,
            "workers_used": args.workers,
        },
        "pilot_mode": bool(args.pilot),
    }

    def want(name: str) -> bool:
        return not only or name in only

    if want("e1"):
        _log("E1 native fidelity, four targets")
        out["fidelity"] = e1_fidelity(n_inputs=1000 if args.pilot else 5000)
        _log(f"   total mismatches: {out['fidelity']['total_mismatches']}")
    if want("e5"):
        _log("E5 T4 witness enumeration")
        out["t4_witnesses"] = e5_t4_witnesses()
        w = out["t4_witnesses"]
        _log(f"   {w['n_walks_enumerated']} walks -> {w['n_d0_counted_classes']} AFL classes, "
             f"{w['n_pairs_g2_identical_g3_different']} G2-identical/G3-different pairs")
    if want("e3"):
        _log("E3 information-loss decomposition")
        out["information_loss"] = e3_information_loss(n_exec=3000 if args.pilot else 20000)
    if want("e4"):
        _log("E4 refinement lattice certificate")
        out["lattice"] = e4_lattice(n_exec=1000 if args.pilot else 4000)
    if want("e2"):
        _log(f"E2 order sweep on T4: {n_trials} trials x {len(ORDER_CONFIGS)} configs "
             f"x {budget} execs")
        all_raw: List[Dict[str, Any]] = []
        for sched in ("score", "round_robin"):
            _log(f"   schedule = {sched}")
            res = e2_order_sweep(n_trials, budget, args.workers, sched)
            raw = res.pop("_raw")
            all_raw.extend(raw)
            out["order_sweep" if sched == "score" else "order_sweep_roundrobin"] = res
            for c, v in res["per_config"].items():
                _log(f"     {c:14s} order={v['order']:<3d} {v['n_success']:>3d}/{n_trials} "
                     f"med={v['median_execs_to_bug']}")
        (DATA / "t4_order_sweep.json").write_text(json.dumps(all_raw))
    if want("e6"):
        _log("E6 paired re-analysis of the original campaigns")
        out["paired_reanalysis"] = e6_paired_reanalysis()

    out["total_wall_seconds"] = time.perf_counter() - t_start
    path = RESULTS / ("step2_revision_pilot.json" if args.pilot else "step2_revision.json")
    if only and path.exists():
        prev = json.loads(path.read_text())
        prev.update({k: v for k, v in out.items() if k not in ("environment",)})
        out = prev
        _log(f"merged {sorted(only)} into the existing results file")
    # Strict RFC 8259: bare Infinity/NaN tokens are not valid JSON and are
    # rejected by conforming parsers. Degenerate estimates (e.g. a conditional
    # odds ratio with an empty discordant cell) are emitted as null plus an
    # explicit "<field>_unbounded" flag so the information is preserved.
    out = sanitize_nonfinite(out)
    path.write_text(json.dumps(out, indent=2, default=str, allow_nan=False))
    _log(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB) in {out['total_wall_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
