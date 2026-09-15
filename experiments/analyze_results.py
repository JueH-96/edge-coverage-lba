#!/usr/bin/env python3
"""Turn trial records into the paper's result documents and summary tables.

Two classes of result document live in ``results/`` after this script runs, and the
distinction matters when reading the output:

**regenerated**
    Recomputed here, in this run, from the raw per-trial records bundled in
    ``data/``.  The scheduler-matrix analysis (survival curves, log-rank and Fisher
    tests, Vargha-Delaney A12 effect sizes, the 192-hypothesis BH-FDR family, and
    the dilution metrics) is regenerated every time, from
    ``data/step2_trials.json``.

**bundled**
    Copied from ``data/precomputed/``.  These come from campaigns whose *analysis*
    cannot be separated from a multi-hour re-execution (the Step 1 validation report
    and the revision experiments both interleave fresh sampling with their
    statistics).  Reviewers who want these regenerated rather than trusted should
    run ``./run_ae.sh --full``, which re-executes the campaigns end to end.

Every entry in the emitted ``results/ae_summary.json`` carries a ``provenance``
field recording which of the two it is, so no number in the summary is presented as
freshly recomputed when it was read from the bundle.

Usage
-----
    python experiments/analyze_results.py            # regenerate + stage + summarise
    python experiments/analyze_results.py --no-regen # stage bundled copies only

Outputs
-------
``results/step2_scheduler_validation.json``  regenerated scheduler analysis
``results/step1_guidance_validation.json``   staged from the bundle
``results/step2_revision.json``              staged from the bundle
``results/corpus_cap_ablation.json``         staged from the bundle
``results/summary_tables.md``                human-readable claim tables
``results/ae_summary.json``                  machine-readable summary + provenance
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

AE_ROOT = Path(__file__).resolve().parent.parent
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))

import ae_paths  # noqa: E402

RESULTS = ae_paths.RESULTS
DATA = ae_paths.DATA
PRE = ae_paths.PRECOMPUTED

#: Documents staged from the bundle when not regenerated in this run.
BUNDLED = {
    "step1_guidance_validation.json": (
        "Step 1 guidance-engine validation: refinement witnesses, partition "
        "discrimination, 540-trial bug-finding campaign, context-depth sweep, "
        "overhead and collision accounting"
    ),
    "step2_revision.json": (
        "Revision experiments: T4 n-gram order sweep, three-way information-loss "
        "decomposition, refinement lattice, paired re-analysis"
    ),
    "corpus_cap_ablation.json": (
        "Fixed-corpus-capacity (K=1000) dilution ablation"
    ),
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


#: Keys whose values are machine properties rather than analysis outputs.
_TIMING_KEYS = ("wall_seconds", "total_wall_seconds", "seconds", "generated_utc",
                "elapsed", "exec_per_s", "execs_per_second")


def _compare_numeric(a: Any, b: Any, path: str = "$",
                     acc: Optional[List[Any]] = None) -> Any:
    """Compare every numeric leaf of two nested structures.

    Integers and booleans must match exactly; floats must agree to a relative
    tolerance of 1e-9, which absorbs platform-dependent floating-point summation
    order without hiding a genuine change in a statistic.

    Returns
    -------
    (int, int, list)
        Number of matching leaves, number of differing leaves, and a list of
        ``(path, regenerated, reference)`` triples for the differences.
    """
    acc = [[0], [0], []] if acc is None else acc
    n_same, n_diff, examples = acc

    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() & b.keys():
            if any(t in k for t in _TIMING_KEYS):
                continue
            _compare_numeric(a[k], b[k], f"{path}.{k}", acc)
    elif isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            _compare_numeric(x, y, f"{path}[{i}]", acc)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a == b:
            n_same[0] += 1
        else:
            n_diff[0] += 1
            examples.append((path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            ok = a == b
        else:
            ok = (math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
                  if math.isfinite(float(a)) and math.isfinite(float(b))
                  else repr(a) == repr(b))
        if ok:
            n_same[0] += 1
        else:
            n_diff[0] += 1
            examples.append((path, a, b))
    return n_same[0], n_diff[0], examples


def _load(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON document, returning ``None`` if it is absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _finite_leaves(obj: Any, path: str = "$", bad: Optional[List[str]] = None
                   ) -> List[str]:
    """Collect the paths of any non-finite float in a nested structure.

    Non-finite floats (``inf``/``nan``) are not representable in RFC 8259 JSON, so
    their presence means the document cannot be read by a conforming parser. The
    Step 2 revision document previously carried bare ``Infinity`` tokens for an
    unbounded odds ratio; that is now serialised as ``null`` plus an explicit
    ``*_unbounded`` flag, and this check guards the regression.
    """
    bad = [] if bad is None else bad
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _finite_leaves(v, f"{path}.{k}", bad)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _finite_leaves(v, f"{path}[{i}]", bad)
    return bad


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def _table_step1(s1: Dict[str, Any]) -> List[str]:
    """Bug-finding success per target x guidance configuration (Step 1)."""
    lines = [
        "### Table 1 - Step 1 bug-finding success (trials with the bug found / trials)",
        "",
        "Blocked design: every configuration sees the same seed sequence, so the "
        "only difference between columns is the feedback function.",
        "",
    ]
    per_target = s1.get("bug_finding", {}).get("per_target", {})
    if not per_target:
        return lines + ["_not available in this results document_", ""]
    configs: List[str] = []
    for t in per_target.values():
        for c in t.get("configs", {}):
            if c not in configs:
                configs.append(c)
    lines.append("| Target | Targeted dim | Budget (execs) | "
                 + " | ".join(configs) + " |")
    lines.append("|---|---|---|" + "---|" * len(configs))
    for name, t in per_target.items():
        cells = []
        for c in configs:
            cfg = t.get("configs", {}).get(c)
            cells.append(f"{cfg['n_bugs_found']}/{cfg['n_trials']}" if cfg else "-")
        lines.append(f"| {name} | {t.get('targeted_dimension', '?')} | "
                     f"{t.get('budget_execs', '?')} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _table_step2(s2: Dict[str, Any]) -> List[str]:
    """Bug-finding success per trigram target x arm (Step 2 campaign matrix)."""
    lines = [
        "### Table 2 - Step 2 energy-schedule matrix (bugs found / trials)",
        "",
        f"Design: {s2.get('n_trials_per_cell', '?')} trials per cell at "
        f"{s2.get('max_execs', '?')} execs, master seed "
        f"{s2.get('master_seed', '?')}, seeds blocked across arms. "
        f"Control = `{s2.get('control_condition', '?')}`, "
        f"treatment = `{s2.get('treatment_condition', '?')}`.",
        "",
    ]
    desc = s2.get("descriptives", {})
    if not desc:
        return lines + ["_not available in this results document_", ""]
    arms: List[str] = []
    for t in desc.values():
        for a in t:
            if a not in arms:
                arms.append(a)
    lines.append("| Target | " + " | ".join(f"`{a}`" for a in arms) + " |")
    lines.append("|---|" + "---|" * len(arms))
    for tname, t in desc.items():
        cells = []
        for a in arms:
            cell = t.get(a)
            cells.append(f"{cell['n_bugs_found']}/{cell['n_trials']}" if cell else "-")
        lines.append(f"| {tname} | " + " | ".join(cells) + " |")
    lines.append("")
    fdr = s2.get("fdr", {})
    if fdr:
        lines += [
            f"BH-FDR over the whole family: {fdr.get('n_hypotheses', '?')} hypotheses, "
            f"{fdr.get('n_rejected', '?')} rejected at q < {fdr.get('alpha', '?')}.",
            "",
        ]
    return lines


def _table_dilution(s2: Dict[str, Any]) -> List[str]:
    """Corpus-dilution metrics: sterile seed fraction and energy Gini per arm."""
    lines = [
        "### Table 3 - Corpus dilution (median over trials)",
        "",
        "`sterile_fraction` is the share of corpus entries that never produced a new "
        "coverage element; `energy_gini` measures how unevenly fuzzing energy is "
        "spread across the corpus. Rising sensitivity should drive Gini up and "
        "sterile fraction down.",
        "",
    ]
    desc = s2.get("descriptives", {})
    if not desc:
        return lines + ["_not available in this results document_", ""]
    lines.append("| Target | Arm | sterile_fraction (median) | energy_gini (median) | "
                 "corpus_size (median) |")
    lines.append("|---|---|---|---|---|")
    for tname, t in desc.items():
        for aname, cell in t.items():
            def med(key: str) -> str:
                blk = cell.get(key)
                if isinstance(blk, dict) and blk.get("median") is not None:
                    return f"{blk['median']:.4g}"
                return "-"
            lines.append(f"| {tname} | `{aname}` | {med('sterile_fraction')} | "
                         f"{med('energy_gini')} | {med('corpus_size')} |")
    lines.append("")
    return lines


def _table_order_sweep(rev: Dict[str, Any]) -> List[str]:
    """T4 n-gram order sweep, round-robin schedule."""
    lines = [
        "### Table 4 - T4 n-gram order sweep (round-robin energy)",
        "",
        "Guidance of order k should discriminate the target exactly when k is at "
        "least the order the bug is built on; edge coverage (`edge_only`) is a "
        "bigram statistic and is predicted to be blind.",
        "",
    ]
    osw = rev.get("order_sweep_roundrobin") or rev.get("order_sweep")
    if not osw:
        return lines + ["_not available in this results document_", ""]
    lines += [
        f"Target `{osw.get('target', '?')}`, schedule `{osw.get('schedule', '?')}`, "
        f"{osw.get('n_trials_per_cell', '?')} trials/cell at "
        f"{osw.get('budget_execs', '?')} execs.",
        "",
        "| Config | n-gram order | successes | success rate | median execs to bug |",
        "|---|---|---|---|---|",
    ]
    for cname, c in osw.get("per_config", {}).items():
        med = c.get("median_execs_to_bug")
        lines.append(
            f"| `{cname}` | {c.get('order', '-')} | "
            f"{c.get('n_success', '-')}/{c.get('n_trials', '-')} | "
            f"{c.get('success_rate', float('nan')):.3f} | "
            f"{med if med is not None else 'n/a (no successes)'} |"
        )
    lines.append("")
    return lines


def _table_fidelity(fid: Optional[Dict[str, Any]]) -> List[str]:
    """Native-C vs Python cross-validation summary."""
    lines = ["### Table 5 - Native-C vs Python model fidelity", ""]
    if not fid:
        return lines + [
            "_not run in this session; run `python tests/test_fidelity.py`_", ""]
    lines += [
        "| Suite | targets | inputs/target | comparisons | mismatches |",
        "|---|---|---|---|---|",
    ]
    for sname, s in fid.get("suites", {}).items():
        lines.append(f"| {sname} | {s['n_targets']} | {s['n_inputs_per_target']} | "
                     f"{s['n_comparisons']} | {s['n_mismatches']} |")
    lines += [
        f"| **total** | {sum(s['n_targets'] for s in fid['suites'].values())} | | "
        f"**{fid['n_comparisons_total']}** | **{fid['n_mismatches_total']}** |",
        "",
    ]
    return lines


def _table_blind_spot(s2: Dict[str, Any]) -> List[str]:
    """The constructive AFL blind-spot witness."""
    bs = s2.get("blind_spot_verification")
    lines = ["### Table 6 - Constructive AFL blind-spot witness", ""]
    if not bs:
        return lines + ["_not available in this results document_", ""]
    lines += [
        f"Target: `{bs.get('target', '?')}`. Two inputs are compared:",
        "",
        "| Property | Identical across the pair? |",
        "|---|---|",
    ]
    for k in ("bigram_multisets_identical", "symbol_counts_identical",
              "d0_walk_maps_identical", "trigram_multisets_identical",
              "d1_signatures_identical"):
        if k in bs:
            lines.append(f"| `{k}` | {'yes' if bs[k] else 'no'} |")
    lines += [
        "",
        f"Outcome: witness A reaches milestone {bs.get('pos_milestone', '?')} "
        f"(bug = {bs.get('pos_bug', '?')}), witness B reaches milestone "
        f"{bs.get('neg_milestone', '?')} (bug = {bs.get('neg_bug', '?')}).",
        "",
    ]
    if bs.get("interpretation"):
        lines += [f"> {bs['interpretation']}", ""]
    return lines


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    """Regenerate what can be regenerated, stage the rest, and summarise."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-regen", action="store_true",
                    help="skip recomputation; stage bundled documents only")
    args = ap.parse_args()

    t0 = time.time()
    ae_paths.ensure_output_dirs()
    provenance: Dict[str, Dict[str, str]] = {}

    # ---- 1. regenerate the scheduler analysis from raw trial records ------- #
    trials_path = DATA / "step2_trials.json"
    if not args.no_regen and trials_path.exists():
        print("[analyze] regenerating scheduler analysis from "
              f"data/{trials_path.name} ...", flush=True)
        sys.path.insert(0, str(AE_ROOT / "experiments"))
        import _analysis_step2 as an
        rc = an.main()
        if rc != 0:
            print("[analyze] FAILED: scheduler analysis returned "
                  f"{rc}", file=sys.stderr)
            return rc
        provenance["step2_scheduler_validation.json"] = {
            "provenance": "regenerated",
            "from": f"data/{trials_path.name}",
            "description": "Step 2 energy-schedule campaign analysis",
        }
    else:
        src = PRE / "step2_scheduler_validation.json"
        if src.exists():
            shutil.copy2(src, RESULTS / src.name)
            provenance[src.name] = {
                "provenance": "bundled",
                "from": f"data/precomputed/{src.name}",
                "description": "Step 2 energy-schedule campaign analysis",
            }

    # ---- 2. stage the documents that require a full re-execution ---------- #
    #
    # An existing results/<name> may be either (a) a copy this script staged on a
    # previous quick run, or (b) genuinely fresh output from a --full run. The two
    # must not be conflated: calling a stale staged copy "regenerated" would
    # overstate what the run actually did. Deciding by content hash against the
    # bundled original settles it without needing any run-to-run state.
    for name, desc in BUNDLED.items():
        dest = RESULTS / name
        src = PRE / name
        if not src.exists():
            if dest.exists():
                provenance[name] = {
                    "provenance": "regenerated",
                    "from": "experiment run in this working tree (no bundled copy "
                            "to compare against)",
                    "description": desc,
                }
            else:
                print(f"[analyze] WARNING: neither results/{name} nor its bundled "
                      "copy exists", flush=True)
            continue

        if dest.exists() and _sha256(dest) != _sha256(src):
            # Differs from the shipped bundle, so a full re-execution produced it.
            provenance[name] = {
                "provenance": "regenerated",
                "from": "--full experiment run in this working tree "
                        "(differs from the bundled copy)",
                "description": desc,
            }
            print(f"[analyze] kept regenerated {name} (differs from bundle)",
                  flush=True)
            continue

        shutil.copy2(src, dest)
        provenance[name] = {"provenance": "bundled",
                            "from": f"data/precomputed/{name}",
                            "description": desc}
        print(f"[analyze] staged bundled {name}", flush=True)

    # ---- 2b. does recomputation reproduce the shipped analysis? ------------ #
    #
    # The scheduler analysis is the one document regenerated from raw trial
    # records on every run, so it can be checked against the copy shipped with the
    # artifact. Agreement means the statistics in the paper follow from the
    # bundled data by the bundled code, which is the specific thing an artifact
    # evaluation is meant to establish.
    recompute_check: Dict[str, Any] = {}
    regen = RESULTS / "step2_scheduler_validation.json"
    ref = PRE / "step2_scheduler_validation.json"
    if provenance.get(regen.name, {}).get("provenance") == "regenerated" \
            and regen.exists() and ref.exists():
        same, diff, examples = _compare_numeric(
            json.loads(regen.read_text()), json.loads(ref.read_text())
        )
        recompute_check = {
            "document": "results/step2_scheduler_validation.json",
            "reference": "data/precomputed/step2_scheduler_validation.json",
            "comparison": "all numeric leaves, exact match on int/bool, "
                          "rel_tol=1e-9 on float",
            "n_numeric_leaves_compared": same + diff,
            "n_matching": same,
            "n_differing": diff,
            "differing_examples": examples[:5],
            "reproduces_reference": diff == 0,
            "note": (
                "Timing fields (wall_seconds and similar) are excluded: they are "
                "properties of the machine, not of the analysis."
            ),
        }
        verdict = "reproduces" if diff == 0 else f"DIFFERS in {diff} value(s)"
        print(f"[analyze] recomputation vs shipped analysis: {verdict} "
              f"({same}/{same + diff} numeric leaves match)", flush=True)

    # ---- 3. strict-JSON / finite-float validation -------------------------- #
    print("[analyze] validating result documents ...", flush=True)
    validation: Dict[str, Any] = {}
    all_ok = True
    for p in sorted(RESULTS.glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            validation[p.name] = {"parses_as_strict_json": False, "error": str(exc)}
            all_ok = False
            continue
        bad = _finite_leaves(doc)
        validation[p.name] = {
            "parses_as_strict_json": True,
            "n_nonfinite_floats": len(bad),
            "nonfinite_paths": bad[:5],
            "size_bytes": p.stat().st_size,
        }
        if bad:
            all_ok = False
        print(f"[analyze]   {p.name:42s} "
              f"{'OK' if not bad else f'{len(bad)} NON-FINITE FLOATS'}", flush=True)

    # ---- 4. summary tables ------------------------------------------------- #
    s1 = _load(RESULTS / "step1_guidance_validation.json") or {}
    s2 = _load(RESULTS / "step2_scheduler_validation.json") or {}
    rev = _load(RESULTS / "step2_revision.json") or {}
    fid = _load(RESULTS / "fidelity_verification.json")

    md: List[str] = [
        "# Artifact-evaluation summary tables",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
        "`experiments/analyze_results.py`.",
        "",
        "Each table names the results document it was read from. Documents marked "
        "**bundled** below were shipped with the artifact rather than recomputed in "
        "this run; see `results/ae_summary.json` for the per-document provenance.",
        "",
        "| Result document | provenance |",
        "|---|---|",
    ]
    for name, info in sorted(provenance.items()):
        md.append(f"| `results/{name}` | {info['provenance']} (from `{info['from']}`) |")
    md.append("")
    if recompute_check:
        md += [
            "### Recomputation check",
            "",
            "`results/step2_scheduler_validation.json` was recomputed here from "
            "`data/step2_trials.json` and compared against the copy shipped with "
            "the artifact:",
            "",
            f"- numeric leaves compared: "
            f"**{recompute_check['n_numeric_leaves_compared']}**",
            f"- matching: **{recompute_check['n_matching']}**, "
            f"differing: **{recompute_check['n_differing']}**",
            f"- verdict: **{'reproduces the shipped analysis' if recompute_check['reproduces_reference'] else 'DIFFERS from the shipped analysis'}**",
            "",
            f"_{recompute_check['note']}_",
            "",
        ]
    md += _table_fidelity(fid)
    md += _table_step1(s1)
    md += _table_step2(s2)
    md += _table_blind_spot(s2)
    md += _table_dilution(s2)
    md += _table_order_sweep(rev)

    (RESULTS / "summary_tables.md").write_text("\n".join(md))
    print(f"[analyze] tables -> results/summary_tables.md ({len(md)} lines)", flush=True)

    # ---- 5. machine-readable summary --------------------------------------- #
    summary: Dict[str, Any] = {
        "schema": "ae_summary/1.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance,
        "recomputation_check": recompute_check or None,
        "json_validation": validation,
        "all_documents_valid": all_ok,
        "headline": {
            "fidelity_comparisons": (fid or {}).get("n_comparisons_total"),
            "fidelity_mismatches": (fid or {}).get("n_mismatches_total"),
            "step2_fdr_hypotheses": s2.get("fdr", {}).get("n_hypotheses"),
            "step2_fdr_rejected": s2.get("fdr", {}).get("n_rejected"),
            "step1_bug_finding_headlines": s1.get("summary", {}).get(
                "bug_finding_headlines"),
            "blind_spot_witness_target": s2.get(
                "blind_spot_verification", {}).get("target"),
        },
        "wall_seconds": round(time.time() - t0, 2),
    }
    (RESULTS / "ae_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False))
    print(f"[analyze] summary -> results/ae_summary.json", flush=True)
    print(f"[analyze] {'OK' if all_ok else 'FAILED'} "
          f"({summary['wall_seconds']}s)", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
