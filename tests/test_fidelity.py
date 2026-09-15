"""Native-C vs Python-model cross-validation: 20,000 exact comparisons, 0 mismatches.

Why this suite exists
---------------------
Every experiment in the paper is executed against *Python* models of the benchmark
targets, because the guidance probes (calling context, value ranges, abstract state)
need to be observed at a granularity that a compiled binary does not expose without
a full instrumentation pass.  That is only scientifically admissible if the Python
models are behaviourally identical to the native C ground truth they stand in for.

This harness establishes that.  For each of the seven benchmark targets it pushes
randomly drawn inputs through both implementations and compares, exactly:

* the full decision trace (the ordered sequence of instrumented branch outcomes),
* the milestone index reached (progress along the target's ladder), and
* the bug flag.

A single divergence on any of the three invalidates the substitution, so the harness
records the first few divergences verbatim rather than only a count.

Budget allocation (20,000 comparisons total)
--------------------------------------------
=========================  =======  ==========  ============
Suite                      targets  per target  comparisons
=========================  =======  ==========  ============
Step 1 (T1-T4)                   4       2,000         8,000
Trigram (TG1-TG3)                3       4,000        12,000
**Total**                        7                **20,000**
=========================  =======  ==========  ============

The trigram targets get the larger share because their input distribution is mixed
(uniform bytes plus perturbations of the key word): uniform bytes essentially never
satisfy a trigram lock, so a uniform-only sample would certify agreement on the
*failing* path only and say nothing about the triggering one.

Run standalone (writes ``results/fidelity_verification.json``)::

    python tests/test_fidelity.py

Run under pytest::

    python -m pytest tests/test_fidelity.py -v
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

AE_ROOT = Path(__file__).resolve().parent.parent
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))

import ae_paths  # noqa: E402
import pytest  # noqa: E402

from test_targets import build_native_lib, check_fidelity  # noqa: E402
from trigram_targets import build_trigram_lib, check_trigram_fidelity  # noqa: E402

# --------------------------------------------------------------------------- #
# Budget: fixed so the headline "20,000 checks" is a property of the harness,
# not of how it happens to be invoked.
# --------------------------------------------------------------------------- #
N_STEP1_PER_TARGET = 2_000
N_TRIGRAM_PER_TARGET = 4_000
N_STEP1_TARGETS = 4        # T1_context, T2_value_range, T3_state_machine, T4_trigram
N_TRIGRAM_TARGETS = 3      # TG1_trigram_lock, TG2_value_trigram, TG3_surprisal_maze
EXPECTED_TOTAL = (
    N_STEP1_PER_TARGET * N_STEP1_TARGETS + N_TRIGRAM_PER_TARGET * N_TRIGRAM_TARGETS
)  # == 20_000

#: Seeds are fixed so the campaign is bit-reproducible across runs and machines.
STEP1_SEED = 20260804
TRIGRAM_SEED = 20260811
MAX_INPUT_LEN = 96

REPORT_PATH = ae_paths.RESULTS / "fidelity_verification.json"


def run_fidelity_campaign(verbose: bool = False) -> Dict[str, Any]:
    """Execute the full 20,000-comparison cross-validation campaign.

    Parameters
    ----------
    verbose : bool, optional
        Print per-stage progress to stdout.

    Returns
    -------
    dict
        Combined report with per-suite detail, the total comparison count, the
        total mismatch count and an overall ``fidelity_ok`` flag.
    """
    t0 = time.time()
    build_native_lib()
    build_trigram_lib()
    if verbose:
        print(f"[fidelity] native libraries built ({time.time() - t0:.1f}s)",
              flush=True)

    if verbose:
        print(f"[fidelity] step 1 suite: {N_STEP1_TARGETS} targets x "
              f"{N_STEP1_PER_TARGET} inputs = "
              f"{N_STEP1_TARGETS * N_STEP1_PER_TARGET} comparisons", flush=True)
    step1 = check_fidelity(
        n_inputs=N_STEP1_PER_TARGET, seed=STEP1_SEED, max_len=MAX_INPUT_LEN
    )

    if verbose:
        print(f"[fidelity] trigram suite: {N_TRIGRAM_TARGETS} targets x "
              f"{N_TRIGRAM_PER_TARGET} inputs = "
              f"{N_TRIGRAM_TARGETS * N_TRIGRAM_PER_TARGET} comparisons", flush=True)
    trigram = check_trigram_fidelity(
        n_inputs=N_TRIGRAM_PER_TARGET, seed=TRIGRAM_SEED, max_len=MAX_INPUT_LEN
    )

    n_step1 = len(step1["per_target"]) * N_STEP1_PER_TARGET
    n_trigram = len(trigram["targets"]) * N_TRIGRAM_PER_TARGET
    total = n_step1 + n_trigram
    mismatches = int(step1["total_mismatches"]) + int(trigram["total_mismatches"])

    report: Dict[str, Any] = {
        "schema": "ae_fidelity_verification/1.0",
        "description": (
            "Exact native-C vs Python-model cross-validation of the decision trace, "
            "milestone index and bug flag for every benchmark target."
        ),
        "n_comparisons_total": total,
        "n_comparisons_expected": EXPECTED_TOTAL,
        "n_mismatches_total": mismatches,
        "fidelity_ok": mismatches == 0 and total == EXPECTED_TOTAL,
        "compiler": step1.get("compiler"),
        "seeds": {"step1": STEP1_SEED, "trigram": TRIGRAM_SEED},
        "max_input_len": MAX_INPUT_LEN,
        "suites": {
            "step1_targets": {
                "n_targets": len(step1["per_target"]),
                "n_inputs_per_target": N_STEP1_PER_TARGET,
                "n_comparisons": n_step1,
                "n_mismatches": int(step1["total_mismatches"]),
                "detail": step1,
            },
            "trigram_targets": {
                "n_targets": len(trigram["targets"]),
                "n_inputs_per_target": N_TRIGRAM_PER_TARGET,
                "n_comparisons": n_trigram,
                "n_mismatches": int(trigram["total_mismatches"]),
                "detail": trigram,
            },
        },
        "wall_seconds": round(time.time() - t0, 2),
    }
    return report


@pytest.fixture(scope="module")
def fidelity_report() -> Dict[str, Any]:
    """Run the campaign once and share it across the assertions below."""
    report = run_fidelity_campaign(verbose=True)
    ae_paths.ensure_output_dirs()
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def test_comparison_budget_is_exactly_20000(fidelity_report) -> None:
    """The campaign must perform the full advertised number of comparisons."""
    assert fidelity_report["n_comparisons_total"] == EXPECTED_TOTAL == 20_000


def test_all_seven_targets_were_covered(fidelity_report) -> None:
    """No target may be silently skipped -- coverage is part of the claim."""
    assert fidelity_report["suites"]["step1_targets"]["n_targets"] == N_STEP1_TARGETS
    assert (
        fidelity_report["suites"]["trigram_targets"]["n_targets"] == N_TRIGRAM_TARGETS
    )


def test_step1_targets_have_zero_mismatches(fidelity_report) -> None:
    """T1-T4 Python models must match the native C library exactly."""
    suite = fidelity_report["suites"]["step1_targets"]
    per_target = suite["detail"]["per_target"]
    offenders = {
        name: info["mismatch_examples"]
        for name, info in per_target.items()
        if info["n_mismatches"]
    }
    assert suite["n_mismatches"] == 0, f"divergences: {offenders}"


def test_trigram_targets_have_zero_mismatches(fidelity_report) -> None:
    """TG1-TG3 Python models must match the native C library exactly."""
    suite = fidelity_report["suites"]["trigram_targets"]
    offenders = {
        name: info["mismatches"]
        for name, info in suite["detail"]["targets"].items()
        if info["n_mismatches"]
    }
    assert suite["n_mismatches"] == 0, f"divergences: {offenders}"


def test_trigram_sample_actually_reaches_the_bug_branch(fidelity_report) -> None:
    """Guard against a vacuous pass.

    If the random sample never drove a target past its first milestone, zero
    mismatches would certify agreement on the trivial rejection path only. At least
    one trigram target must reach a milestone beyond the first for the result to
    carry weight.
    """
    targets = fidelity_report["suites"]["trigram_targets"]["detail"]["targets"]
    max_milestones = {n: info["max_milestone_seen"] for n, info in targets.items()}
    assert max(max_milestones.values()) >= 2, (
        "fidelity sample never progressed past milestone 1; the 0-mismatch result "
        f"would be vacuous. max milestone per target: {max_milestones}"
    )


def test_overall_fidelity_flag(fidelity_report) -> None:
    """The single headline claim: 20,000/20,000 exact agreements."""
    assert fidelity_report["fidelity_ok"] is True
    assert fidelity_report["n_mismatches_total"] == 0


def main() -> int:
    """Standalone entry point: run the campaign and write the JSON report."""
    report = run_fidelity_campaign(verbose=True)
    ae_paths.ensure_output_dirs()
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False))
    ok = report["fidelity_ok"]
    print(
        f"[fidelity] {report['n_comparisons_total']}/"
        f"{report['n_comparisons_expected']} comparisons, "
        f"{report['n_mismatches_total']} mismatches -> "
        f"{'PASS' if ok else 'FAIL'}  ({report['wall_seconds']}s)",
        flush=True,
    )
    print(f"[fidelity] report -> {REPORT_PATH.relative_to(AE_ROOT)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
