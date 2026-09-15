"""
Minimal mutational fuzzing harness with a pluggable feedback oracle.

The purpose of this harness is *experimental control*. Every configuration shares
the same mutation operators, the same seed corpus, the same per-trial RNG seed, the
same round-robin scheduler and the same execution budget. The **only** thing that
differs between configurations is which dimensions of the guidance engine are
enabled, i.e. the feedback function that decides corpus admission. Any difference in
outcome is therefore attributable to the coverage metric and not to a scheduling or
mutation confound.

Two deliberate design decisions
-------------------------------
**Round-robin scheduling with fixed energy.** The engine computes a fused novelty
score, but this harness does *not* use it for scheduling. Energy assignment is a
Step-2 concern; mixing it in here would introduce a second difference between
configurations. A consequence is that richer coverage metrics admit more corpus
entries and therefore mutate each entry less often - *corpus dilution*. That is a
genuine cost of added sensitivity, and round-robin makes it visible rather than
hiding it behind a scheduler that compensates for it.

**Blocked (paired) trial seeds.** Trial ``i`` uses the same seed for every
configuration, so the comparison is a randomized block design blocked on seed. This
supports paired analysis (Wilcoxon signed-rank) alongside the unpaired test
(Mann-Whitney U) recommended for fuzzing experiments.

Statistical protocol follows Klees et al., "Evaluating Fuzz Testing" (CCS 2018):
many independent trials, a statistical test rather than a single run, an effect size
(Vargha-Delaney A12), and explicit treatment of trials that never find the bug
(right-censored at the budget).
"""

from __future__ import annotations

import heapq
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from guidance_engine import MultiDimGuidanceEngine  # noqa: E402
from test_targets import get_targets  # noqa: E402

__all__ = [
    "CONFIGS",
    "CONFIG_PARAMS",
    "BLIND_CONFIG",
    "Mutator",
    "TrialRecord",
    "run_trial",
    "worker",
    "trial_seed",
    "SEED_INPUT",
    "SEED_INPUT_BY_TARGET",
    "MAX_INPUT_LEN",
    "TARGET_BUDGETS",
]

#: Per-target execution budgets, calibrated from pilot runs so that the strongest
#: configuration reaches the seeded bug in a majority of trials while the baseline
#: still has a fair chance. The budget is identical across configurations for a
#: given target, which is what the comparison requires.
TARGET_BUDGETS: Dict[str, int] = {
    "T1_context": 60_000,
    "T2_value_range": 300_000,
    "T3_state_machine": 60_000,
    "T4_trigram": 150_000,
}

#: Ablation configurations. ``blind_random`` is a feedback-free lower bound (the
#: sanity baseline Klees et al. recommend), ``edge_only`` is the standard AFL-style
#: 2D edge-coverage control, each ``d0_dN`` adds exactly one guidance dimension, and
#: ``full_3d`` is the complete 3D engine.
CONFIGS: Dict[str, Tuple[str, ...]] = {
    "blind_random": ("D0",),
    "edge_only": ("D0",),
    "d0_d1_context": ("D0", "D1"),
    "d0_d2_valuerange": ("D0", "D2"),
    "d0_d3_statemachine": ("D0", "D3"),
    "full_3d": ("D0", "D1", "D2", "D3"),
    # ---- order sweep (T4): competing published abstractions at fixed order ----
    "ngram2": ("DN",),          # == AFL edge coverage, by construction
    "ngram3": ("DN",),
    "ngram4": ("DN",),
    "ngram6": ("DN",),
    "ngram8": ("DN",),
    "ctx2": ("D0", "D1"),
    "ctx3": ("D0", "D1"),
    "ctx4": ("D0", "D1"),
    "ctx6": ("D0", "D1"),
    "ctx8": ("D0", "D1"),
    "ctx16": ("D0", "D1"),
}

#: Per-configuration engine parameter overrides. Everything not listed here uses
#: the campaign-wide defaults, so a configuration differs from the control in
#: exactly the dimensions and the single order parameter named below.
CONFIG_PARAMS: Dict[str, Dict[str, int]] = {
    "ngram2": {"ngram": 2},
    "ngram3": {"ngram": 3},
    "ngram4": {"ngram": 4},
    "ngram6": {"ngram": 6},
    "ngram8": {"ngram": 8},
    "ctx2": {"context_depth": 2},
    "ctx3": {"context_depth": 3},
    "ctx4": {"context_depth": 4},
    "ctx6": {"context_depth": 6},
    "ctx8": {"context_depth": 8},
    "ctx16": {"context_depth": 16},
}

#: Configuration whose coverage feedback is discarded: the corpus never grows, so
#: every input is a fresh mutation of the seed. Measures blind-search difficulty
#: in-situ under exactly the same mutation operators and budget.
BLIND_CONFIG = "blind_random"

#: Uninformative seed input shared by every configuration and target.
SEED_INPUT = bytes(32)
#: Per-target seed override. The seed corpus is part of the *harness*, not of the
#: coverage metric, and is held identical across configurations; sizing it to the
#: input format is what a user of any fuzzer would do. T4 reads a fixed six-step
#: walk, so a 32-byte seed would leave 26 bytes that no configuration can ever
#: use and would inflate every arm's mutation cost by the same factor, wasting
#: budget without changing the comparison.
SEED_INPUT_BY_TARGET: Dict[str, bytes] = {"T4_trigram": bytes(6)}
MAX_INPUT_LEN = 256


# --------------------------------------------------------------------------- #
# Mutator
# --------------------------------------------------------------------------- #
class Mutator:
    """AFL-style havoc mutator, fully determined by its RNG.

    Implements the mutation families AFL uses in its havoc stage: bit flips,
    interesting-value substitution at 8/16/32-bit widths, small arithmetic deltas,
    random byte writes, chunk deletion, chunk cloning/insertion, chunk overwrite,
    and two-parent splicing.

    Parameters
    ----------
    rng : random.Random
        Seeded RNG; all randomness flows from here so trials are reproducible.
    max_len : int, optional
        Hard cap on generated input length, default :data:`MAX_INPUT_LEN`.
    """

    INTERESTING_8 = (0x00, 0x01, 0x02, 0x10, 0x20, 0x40, 0x64, 0x7F, 0x80, 0xFF)
    INTERESTING_16 = (0x0000, 0x0001, 0x0080, 0x00FF, 0x0100, 0x0200, 0x03E8,
                      0x0400, 0x1000, 0x7FFF, 0x8000, 0xFFFF)
    ARITH_MAX = 35

    def __init__(self, rng: random.Random, max_len: int = MAX_INPUT_LEN) -> None:
        self.rng = rng
        self.max_len = int(max_len)

    def mutate(self, data: bytes, splice_pool: Optional[Sequence[bytes]] = None) -> bytes:
        """Return a mutated copy of ``data``.

        Parameters
        ----------
        data : bytes
            Parent input.
        splice_pool : sequence of bytes, optional
            Other corpus entries available for splicing.

        Returns
        -------
        bytes
            Mutated input, length clamped to ``[1, max_len]``.
        """
        rng = self.rng
        buf = bytearray(data) if data else bytearray(1)
        n_ops = 1 << rng.randrange(0, 5)  # 1, 2, 4, 8 or 16 stacked mutations
        for _ in range(n_ops):
            if not buf:
                buf = bytearray(1)
            op = rng.randrange(0, 10)
            ln = len(buf)
            if op == 0:  # bit flip
                i = rng.randrange(ln)
                buf[i] ^= 1 << rng.randrange(8)
            elif op == 1:  # interesting 8-bit value
                buf[rng.randrange(ln)] = rng.choice(self.INTERESTING_8)
            elif op == 2:  # interesting 16-bit value, random endianness
                if ln >= 2:
                    i = rng.randrange(ln - 1)
                    v = rng.choice(self.INTERESTING_16)
                    if rng.random() < 0.5:
                        buf[i] = v & 0xFF
                        buf[i + 1] = (v >> 8) & 0xFF
                    else:
                        buf[i] = (v >> 8) & 0xFF
                        buf[i + 1] = v & 0xFF
            elif op == 3:  # interesting 32-bit value
                if ln >= 4:
                    i = rng.randrange(ln - 3)
                    v = rng.getrandbits(32) if rng.random() < 0.3 else rng.choice(
                        (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
                    )
                    for k in range(4):
                        buf[i + k] = (v >> (8 * k)) & 0xFF
            elif op == 4:  # small arithmetic delta
                i = rng.randrange(ln)
                d = rng.randint(1, self.ARITH_MAX)
                if rng.random() < 0.5:
                    d = -d
                buf[i] = (buf[i] + d) & 0xFF
            elif op == 5:  # random byte
                buf[rng.randrange(ln)] = rng.getrandbits(8)
            elif op == 6:  # delete a chunk
                if ln > 2:
                    clen = rng.randint(1, max(1, min(ln - 1, ln // 4)))
                    i = rng.randrange(ln - clen + 1)
                    del buf[i : i + clen]
            elif op == 7:  # clone or insert a chunk
                if ln < self.max_len:
                    clen = rng.randint(1, max(1, min(16, self.max_len - ln)))
                    i = rng.randrange(ln + 1)
                    if rng.random() < 0.75 and ln >= clen:
                        src = rng.randrange(ln - clen + 1)
                        chunk = bytes(buf[src : src + clen])
                    else:
                        chunk = bytes(rng.getrandbits(8) for _ in range(clen))
                    buf[i:i] = chunk
            elif op == 8:  # overwrite a chunk with another chunk
                if ln >= 4:
                    clen = rng.randint(1, max(1, ln // 4))
                    dst = rng.randrange(ln - clen + 1)
                    src = rng.randrange(ln - clen + 1)
                    buf[dst : dst + clen] = buf[src : src + clen]
            else:  # splice with another corpus entry
                if splice_pool:
                    other = splice_pool[rng.randrange(len(splice_pool))]
                    if other and len(other) >= 2 and ln >= 2:
                        cut_a = rng.randrange(1, ln)
                        cut_b = rng.randrange(1, len(other))
                        buf = bytearray(bytes(buf[:cut_a]) + other[cut_b:])
        if len(buf) > self.max_len:
            del buf[self.max_len :]
        if not buf:
            buf = bytearray(1)
        return bytes(buf)


# --------------------------------------------------------------------------- #
# Trial
# --------------------------------------------------------------------------- #
def trial_seed(master_seed: int, target_name: str, trial_index: int) -> int:
    """Deterministic per-trial seed, identical across configurations.

    Blocking on the seed makes the configuration comparison a randomized block
    design and permits paired analysis.

    Parameters
    ----------
    master_seed : int
        Master seed for the whole experiment.
    target_name : str
        Target identifier (so different targets get different seed streams).
    trial_index : int
        Index of the trial.

    Returns
    -------
    int
        Seed value.
    """
    h = 1469598103934665603
    for ch in f"{master_seed}|{target_name}|{trial_index}":
        h = ((h ^ ord(ch)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h & 0x7FFFFFFF


@dataclass
class TrialRecord:
    """Outcome of one fuzzing trial.

    Attributes
    ----------
    first_bug_exec : int or None
        Execution index at which the seeded bug first fired; ``None`` means the
        trial is right-censored at ``execs``.
    milestone_first_exec : dict
        Execution index at which each milestone level was first reached.
    curve : list of tuple
        Sampled ``(execs, exact_elements, corpus_size, max_milestone)``.
    """

    target: str = ""
    config: str = ""
    seed: int = 0
    execs: int = 0
    wall_seconds: float = 0.0
    corpus_size: int = 0
    n_interesting: int = 0
    first_bug_exec: Optional[int] = None
    n_bugs: int = 0
    max_milestone: int = 0
    milestone_first_exec: Dict[int, int] = field(default_factory=dict)
    curve: List[Tuple[int, int, int, int]] = field(default_factory=list)
    engine_stats: Dict[str, Any] = field(default_factory=dict)


def run_trial(
    target_name: str,
    config_name: str,
    seed: int,
    max_execs: int = 20000,
    curve_every: int = 500,
    stop_on_bug: bool = True,
    map_bits: int = 16,
    context_depth: int = 16,
    kgram: int = 3,
    ngram: int = 3,
    track_exact: bool = True,
    schedule: str = "round_robin",
) -> TrialRecord:
    """Run one independent fuzzing trial.

    Parameters
    ----------
    target_name : str
        Benchmark target name.
    config_name : str
        Key of :data:`CONFIGS`.
    seed : int
        RNG seed (use :func:`trial_seed` for blocked seeds).
    max_execs : int, optional
        Execution budget, default 20000.
    curve_every : int, optional
        Sampling interval for the coverage curve, default 500.
    stop_on_bug : bool, optional
        Stop the trial at the first bug (time-to-first-bug semantics), default True.
    map_bits, context_depth, kgram : int, optional
        Guidance-engine parameters, held constant across configurations.
    track_exact : bool, optional
        Maintain exact element sets for analysis, default True.
    schedule : {"round_robin", "score"}, optional
        Seed-selection policy. ``round_robin`` cycles the corpus with equal
        energy and is the default used everywhere in the first version of this
        study, because it introduces no second difference between configurations.
        ``score`` selects the entry maximising ``s / (1 + n_fuzz)``, where ``s``
        is the engine's fused novelty score for the execution that admitted the
        entry and ``n_fuzz`` counts how often it has already been mutated. The
        rule is the least-fuzzed-first policy familiar from AFL's queue
        management, weighted by the novelty score the engine already computes
        and that no other experiment in this paper consumes. It is included as an
        explicit experimental factor because the cost model of the theory section
        predicts that the *scheduler*, not the abstraction, is what bounds the
        benefit of higher-order feedback.

    Returns
    -------
    TrialRecord
    """
    targets = {t.NAME: t for t in get_targets()}
    target = targets[target_name]
    dims = CONFIGS[config_name]
    use_feedback = config_name != BLIND_CONFIG
    over = CONFIG_PARAMS.get(config_name, {})
    context_depth = int(over.get("context_depth", context_depth))
    ngram = int(over.get("ngram", ngram))

    rng = random.Random(seed)
    mut = Mutator(rng)
    engine = MultiDimGuidanceEngine(
        dims=dims,
        map_bits=map_bits,
        context_depth=context_depth,
        kgram=kgram,
        ngram=ngram,
        track_exact=track_exact,
    )

    if schedule not in ("round_robin", "score"):
        raise ValueError(f"unknown schedule: {schedule}")
    seed_input = SEED_INPUT_BY_TARGET.get(target_name, SEED_INPUT)
    rec = TrialRecord(target=target_name, config=config_name, seed=seed)
    corpus: List[bytes] = [seed_input]
    # score-schedule bookkeeping: a max-heap on -score/(1+n_fuzz)
    heap: List[Tuple[float, int, int]] = []
    n_fuzz: List[int] = [0]
    scores: List[float] = [1.0]
    if schedule == "score":
        heapq.heappush(heap, (-1.0, 0, 0))
    t0 = time.perf_counter()

    # Prime the engine with the seed input so the initial coverage is claimed.
    engine.reset()
    target.run(seed_input, engine)
    engine.evaluate(commit=True)

    cursor = 0
    execs = 0
    while execs < max_execs:
        if schedule == "score":
            _, _, pi = heapq.heappop(heap)
            n_fuzz[pi] += 1
            heapq.heappush(heap, (-scores[pi] / (1 + n_fuzz[pi]), n_fuzz[pi], pi))
            parent = corpus[pi]
        else:
            parent = corpus[cursor % len(corpus)]
            cursor += 1
        data = mut.mutate(parent, splice_pool=corpus)

        engine.reset()
        res = target.run(data, engine)
        fb = engine.evaluate(commit=True)
        execs += 1

        if res.milestone > rec.max_milestone:
            for lvl in range(rec.max_milestone + 1, res.milestone + 1):
                rec.milestone_first_exec[lvl] = execs
            rec.max_milestone = res.milestone

        if res.bug:
            rec.n_bugs += 1
            if rec.first_bug_exec is None:
                rec.first_bug_exec = execs
            if stop_on_bug:
                break

        if fb.is_interesting:
            rec.n_interesting += 1
            if use_feedback:
                corpus.append(data)
                if schedule == "score":
                    scores.append(max(fb.fused_score, 1e-6))
                    n_fuzz.append(0)
                    heapq.heappush(heap, (-scores[-1], 0, len(corpus) - 1))

        if execs % curve_every == 0:
            st = engine.stats()
            rec.curve.append(
                (execs, int(st["total_exact_elements"]), len(corpus), rec.max_milestone)
            )

    rec.execs = execs
    rec.wall_seconds = time.perf_counter() - t0
    rec.corpus_size = len(corpus)
    st = engine.stats()
    rec.engine_stats = {
        "total_exact_elements": int(st["total_exact_elements"]),
        "per_dimension": {
            d: {
                k: v
                for k, v in s.items()
                if k
                in (
                    "exact_elements",
                    "mapped_slots_covered",
                    "collision_rate",
                    "distinct_contexts_exact",
                    "n_abstract_states",
                    "n_transitions",
                    "n_solved_comparisons",
                    "mean_node_entropy_bits",
                    "ngram_k",
                    "context_depth",
                )
            }
            for d, s in st["per_dimension"].items()
        },
    }
    if not rec.curve or rec.curve[-1][0] != execs:
        rec.curve.append(
            (execs, int(st["total_exact_elements"]), len(corpus), rec.max_milestone)
        )
    return rec


def worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Multiprocessing entry point: run one trial and return a plain dict.

    Parameters
    ----------
    job : dict
        Keyword arguments for :func:`run_trial`, optionally plus a ``"tag"`` key
        that is echoed back in the result (used to label sweep arms).

    Returns
    -------
    dict
        JSON-serialisable trial record.
    """
    job = dict(job)
    tag = job.pop("tag", None)
    rec = run_trial(**job)
    out = {
        "target": rec.target,
        "config": rec.config,
        "seed": rec.seed,
        "execs": rec.execs,
        "wall_seconds": rec.wall_seconds,
        "corpus_size": rec.corpus_size,
        "n_interesting": rec.n_interesting,
        "first_bug_exec": rec.first_bug_exec,
        "n_bugs": rec.n_bugs,
        "max_milestone": rec.max_milestone,
        "milestone_first_exec": {str(k): v for k, v in rec.milestone_first_exec.items()},
        "curve": rec.curve,
        "engine_stats": rec.engine_stats,
    }
    if tag is not None:
        out["tag"] = tag
    return out
