"""Step 2 fuzzing harness: energy-scheduled trials over the trigram targets.

Differs from the Step 1 harness (``fuzz_harness.py``) in exactly one structural
respect, which is the whole point of Step 2: a trial no longer spends one
mutation per queue entry per cycle. It *selects* an entry and grants it a burst
of ``p_s`` mutations - its energy - before re-selecting. Everything else (the
havoc mutator, the blocked seeding, the right-censoring semantics, the guidance
engine) is reused unchanged from Step 1, so a difference between a Step 1 and a
Step 2 condition is attributable to the schedule and not to the plumbing.

The mutator is imported from ``fuzz_harness`` rather than reimplemented,
deliberately: a second mutator would be a second uncontrolled difference between
the two steps and would make the Step 1 round-robin numbers non-comparable.
"""

from __future__ import annotations

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

SESSION = AE_ROOT
# --- end bootstrap --------------------------------------------------------- #
if str(AE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(AE_ROOT / "src"))

from energy_scheduler import (  # noqa: E402
    CorpusEntry,
    DEPTH_LADDER,
    DynamicContextManager,
    make_schedule,
)
from fuzz_harness import Mutator, trial_seed  # noqa: E402
from guidance_engine import CallingContextTracker, MultiDimGuidanceEngine  # noqa: E402
from trigram_targets import SEED_INPUT_BY_TARGET, get_trigram_targets  # noqa: E402

__all__ = [
    "STEP2_CONFIGS",
    "Step2Config",
    "Step2Record",
    "run_step2_trial",
    "step2_worker",
    "trial_seed",
]


# --------------------------------------------------------------------------- #
# Experimental conditions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step2Config:
    """One experimental condition: a set of dimensions plus a power schedule.

    Attributes
    ----------
    name : str
        Condition label used throughout the results.
    dims : tuple of str
        Guidance dimensions enabled in the engine.
    schedule : str
        Key of :data:`energy_scheduler.SCHEDULES`.
    context_depth : int
        Fixed D1 window depth; ignored when ``dynamic_depth`` is True.
    dynamic_depth : bool
        Whether :class:`DynamicContextManager` drives the D1 window.
    use_feedback : bool
        False only for the blind control, which never admits corpus entries and
        therefore measures what the mutator alone achieves.
    description : str
        One-line rationale, carried into the results record.
    """

    name: str
    dims: Tuple[str, ...]
    schedule: str
    context_depth: int = 4
    dynamic_depth: bool = False
    use_feedback: bool = True
    description: str = ""


#: The experimental matrix. Every 3D condition uses the *same* dimension set, so
#: the only thing that varies across conditions 3-8 is the power schedule and the
#: context-depth policy - which is what makes the contrast interpretable.
#:
#: Fixed-depth 3D conditions run at N = 4 because Step 1 found N = 4 to be the
#: best fixed depth; beating a tuned baseline is the claim worth making. The
#: N = 16 condition is retained separately to reproduce the Step 1 dilution
#: effect under the new harness rather than citing it.
STEP2_CONFIGS: Tuple[Step2Config, ...] = (
    Step2Config(
        "blind_random",
        ("D0",),
        "uniform_rr",
        use_feedback=False,
        description="No feedback: mutator-only control. Bounds what luck achieves.",
    ),
    Step2Config(
        "afl_d0_rr",
        ("D0",),
        "uniform_rr",
        description="AFL edge coverage, uniform round robin. The Step 1 baseline.",
    ),
    Step2Config(
        "afl_d0_fast",
        ("D0",),
        "afl_fast",
        description="AFL edge coverage with the AFLFast FAST power schedule "
        "(Bohme et al. CCS'16). Strongest single-dimensional baseline.",
    ),
    Step2Config(
        "d3d_rr_n4",
        ("D0", "D1", "D2", "D3"),
        "uniform_rr",
        context_depth=4,
        description="3D guidance under Step 1 round robin at the Step 1 optimal "
        "depth. The condition Step 2 must beat.",
    ),
    Step2Config(
        "d3d_rr_n16",
        ("D0", "D1", "D2", "D3"),
        "uniform_rr",
        context_depth=16,
        description="3D guidance under round robin at depth 16: the Step 1 "
        "dilution condition, reproduced under the Step 2 harness.",
    ),
    Step2Config(
        "d3d_fast_n4",
        ("D0", "D1", "D2", "D3"),
        "afl_fast",
        context_depth=4,
        description="3D guidance with the FAST power schedule: does a "
        "single-dimensional schedule suffice once the feedback is multidimensional?",
    ),
    Step2Config(
        "d3d_static_n4",
        ("D0", "D1", "D2", "D3"),
        "static_3d",
        context_depth=4,
        description="Fixed-weight fused-novelty energy. Isolates 'consult the "
        "dimensions' from 'adapt to them'.",
    ),
    Step2Config(
        "d3d_adaptive_n4",
        ("D0", "D1", "D2", "D3"),
        "adaptive_3d",
        context_depth=4,
        description="Adaptive 3D energy schedule at fixed depth 4.",
    ),
    Step2Config(
        "d3d_adaptive_dyn",
        ("D0", "D1", "D2", "D3"),
        "adaptive_3d",
        context_depth=4,
        dynamic_depth=True,
        description="Adaptive 3D energy schedule with dynamic context depth: "
        "starts at N = 4 and escalates on novelty saturation.",
    ),
)

CONFIG_BY_NAME: Dict[str, Step2Config] = {c.name: c for c in STEP2_CONFIGS}


# --------------------------------------------------------------------------- #
# Trial record
# --------------------------------------------------------------------------- #
@dataclass
class Step2Record:
    """Outcome of one Step 2 trial.

    Attributes
    ----------
    first_bug_exec : int or None
        Execution index at which the seeded bug first fired. ``None`` marks a
        right-censored trial (budget exhausted), which the log-rank test
        consumes as a censoring indicator rather than as a missing value.
    milestone_first_exec : dict
        Execution index at which each milestone level was first reached.
    curve : list of list
        Sampled ``[execs, exact_elements, corpus_size, max_milestone,
        context_depth]``.
    """

    target: str = ""
    config: str = ""
    seed: int = 0
    execs: int = 0
    wall_seconds: float = 0.0
    execs_per_second: float = 0.0
    corpus_size: int = 0
    n_interesting: int = 0
    first_bug_exec: Optional[int] = None
    n_bugs: int = 0
    max_milestone: int = 0
    milestone_first_exec: Dict[int, int] = field(default_factory=dict)
    curve: List[List[int]] = field(default_factory=list)
    engine_stats: Dict[str, Any] = field(default_factory=dict)
    schedule_stats: Dict[str, Any] = field(default_factory=dict)
    depth_stats: Dict[str, Any] = field(default_factory=dict)
    energy_hist: Dict[str, int] = field(default_factory=dict)


def _path_key(engine: MultiDimGuidanceEngine) -> int:
    """Stable hash of the current D0 coverage signature.

    AFLFast's ``f(i)`` counts how often the *path* an entry exercises has been
    fuzzed, so entries that rediscover a well-trodden path share a counter. Using
    the D0 signature rather than the fused signature keeps ``f(i)`` meaning the
    same thing across conditions with different dimension sets - otherwise the
    FAST baseline would be handed a different (and finer) notion of "path" purely
    as a side effect of enabling D1.
    """
    try:
        return hash(engine.d0.trace_signature())
    except Exception:  # pragma: no cover - D0 always enabled in this study
        return 0


def run_step2_trial(
    target_name: str,
    config_name: str,
    seed: int,
    max_execs: int = 20000,
    curve_every: int = 500,
    stop_on_bug: bool = True,
    map_bits: int = 16,
    kgram: int = 3,
    track_exact: bool = True,
    depth_window: int = 500,
    depth_saturation_ratio: float = 0.05,
    depth_patience: int = 2,
    depth_min_execs: int = 1000,
) -> Step2Record:
    """Run one independent, energy-scheduled fuzzing trial.

    Parameters
    ----------
    target_name : str
        One of the trigram target names.
    config_name : str
        Key of :data:`CONFIG_BY_NAME`.
    seed : int
        RNG seed; use :func:`trial_seed` so conditions share a seed block.
    max_execs : int, optional
        Execution budget, default 20000. Trials that exhaust it are
        right-censored.
    curve_every : int, optional
        Sampling interval for the trajectory curve, default 500.
    stop_on_bug : bool, optional
        Stop at the first bug (time-to-first-bug semantics), default True.
    map_bits, kgram : int, optional
        Guidance-engine parameters, held constant across all conditions.
    track_exact : bool, optional
        Maintain exact element sets, default True.
    depth_window, depth_saturation_ratio, depth_patience, depth_min_execs
        :class:`DynamicContextManager` parameters; used only when the condition
        enables dynamic depth.

    Returns
    -------
    Step2Record
    """
    cfg = CONFIG_BY_NAME[config_name]
    targets = {t.NAME: t for t in get_trigram_targets()}
    target = targets[target_name]

    rng = random.Random(seed)
    mut = Mutator(rng)

    ctx_mgr: Optional[DynamicContextManager] = None
    depth = cfg.context_depth
    if cfg.dynamic_depth:
        ctx_mgr = DynamicContextManager(
            ladder=DEPTH_LADDER,
            start_depth=cfg.context_depth,
            window=depth_window,
            saturation_ratio=depth_saturation_ratio,
            patience=depth_patience,
            min_execs_at_depth=depth_min_execs,
        )
        depth = ctx_mgr.depth

    engine = MultiDimGuidanceEngine(
        dims=cfg.dims,
        map_bits=map_bits,
        context_depth=depth,
        kgram=kgram,
        track_exact=track_exact,
    )
    sched = make_schedule(cfg.schedule, rng, cfg.dims)

    seed_input = SEED_INPUT_BY_TARGET[target_name]
    rec = Step2Record(target=target_name, config=config_name, seed=seed)
    corpus_bytes: List[bytes] = [seed_input]

    t0 = time.perf_counter()

    # Prime the engine on the seed input so its coverage is claimed, then admit
    # the seed as corpus entry 0 with whatever novelty it produced.
    engine.reset()
    target.run(seed_input, engine)
    fb0 = engine.evaluate(commit=True)
    sched.admit(
        CorpusEntry(
            index=0,
            data=seed_input,
            novelty={d: fb0.new_elements(d) for d in cfg.dims},
            surprisal_bits=float(fb0.per_dim.get("D3", {}).get("surprisal_bits", 0.0)),
            fused_score=float(fb0.fused_score),
            path_key=_path_key(engine),
            depth_at_admission=depth,
            admitted_exec=0,
        )
    )

    execs = 0
    while execs < max_execs:
        idx, energy = sched.select(execs)
        parent = sched.entries[idx].data
        n_children = 0
        dim_yield: Dict[str, int] = {d: 0 for d in cfg.dims}
        used = 0
        hit_bug = False

        for _ in range(energy):
            if execs >= max_execs:
                break
            data = mut.mutate(parent, splice_pool=corpus_bytes)

            engine.reset()
            res = target.run(data, engine)
            fb = engine.evaluate(commit=True)
            execs += 1
            used += 1

            if res.milestone > rec.max_milestone:
                for lvl in range(rec.max_milestone + 1, res.milestone + 1):
                    rec.milestone_first_exec[lvl] = execs
                rec.max_milestone = res.milestone

            if res.bug:
                rec.n_bugs += 1
                if rec.first_bug_exec is None:
                    rec.first_bug_exec = execs
                if stop_on_bug:
                    hit_bug = True
                    break

            for d in cfg.dims:
                dim_yield[d] += fb.new_elements(d)

            if fb.is_interesting:
                rec.n_interesting += 1
                if cfg.use_feedback:
                    corpus_bytes.append(data)
                    sched.admit(
                        CorpusEntry(
                            index=0,  # assigned by admit()
                            data=data,
                            novelty={d: fb.new_elements(d) for d in cfg.dims},
                            surprisal_bits=float(
                                fb.per_dim.get("D3", {}).get("surprisal_bits", 0.0)
                            ),
                            fused_score=float(fb.fused_score),
                            path_key=_path_key(engine),
                            depth_at_admission=depth,
                            admitted_exec=execs,
                        )
                    )
                    n_children += 1

            # -- dynamic context depth ------------------------------------- #
            if ctx_mgr is not None:
                n_new_ctx = int(fb.per_dim.get("D1", {}).get("new_contexts", 0))
                if ctx_mgr.observe(n_new_ctx):
                    depth = ctx_mgr.depth
                    # Rebuild D1 at the new depth. Sound because the depth-N
                    # context partition refines the depth-N' one for N > N', so
                    # re-learning re-derives every distinction the coarser window
                    # held plus new ones; nothing previously claimed is silently
                    # lost. The re-learning cost is exactly why escalation is
                    # gated on saturation instead of applied from the start.
                    engine.d1 = CallingContextTracker(
                        depth=depth,
                        map_bits=map_bits,
                        recursion_cap=engine.d1.recursion_cap,
                        track_exact=track_exact,
                    )

            if execs % curve_every == 0:
                st = engine.stats()
                rec.curve.append(
                    [
                        execs,
                        int(st["total_exact_elements"]),
                        len(sched.entries),
                        rec.max_milestone,
                        depth,
                    ]
                )

        sched.report(idx, n_children, dim_yield, used)
        if hit_bug:
            break

    rec.execs = execs
    rec.wall_seconds = time.perf_counter() - t0
    rec.execs_per_second = execs / rec.wall_seconds if rec.wall_seconds > 0 else 0.0
    rec.corpus_size = len(sched.entries)
    rec.schedule_stats = sched.stats()
    if ctx_mgr is not None:
        rec.depth_stats = ctx_mgr.stats()

    # Energy histogram over log2 buckets: the shape of the allocation, without
    # shipping one row per selection back from the worker.
    hist: Dict[str, int] = {}
    for e in sched.entries:
        if e.energy_spent <= 0:
            key = "0"
        else:
            key = str(1 << max(0, e.energy_spent.bit_length() - 1))
        hist[key] = hist.get(key, 0) + 1
    rec.energy_hist = hist

    st = engine.stats()
    rec.engine_stats = {
        "total_exact_elements": int(st["total_exact_elements"]),
        "final_context_depth": depth,
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
                    "context_depth",
                )
            }
            for d, s in st["per_dimension"].items()
        },
    }
    if not rec.curve or rec.curve[-1][0] != execs:
        rec.curve.append(
            [
                execs,
                int(st["total_exact_elements"]),
                len(sched.entries),
                rec.max_milestone,
                depth,
            ]
        )
    return rec


def step2_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Multiprocessing entry point: run one trial, return a plain dict.

    Parameters
    ----------
    job : dict
        ``target``, ``config``, ``seed``, ``max_execs``, ``curve_every``.

    Returns
    -------
    dict
        Serialisable :class:`Step2Record` fields.
    """
    rec = run_step2_trial(
        target_name=job["target"],
        config_name=job["config"],
        seed=job["seed"],
        max_execs=job.get("max_execs", 20000),
        curve_every=job.get("curve_every", 500),
    )
    d = dict(rec.__dict__)
    d["milestone_first_exec"] = {str(k): v for k, v in rec.milestone_first_exec.items()}
    return d
