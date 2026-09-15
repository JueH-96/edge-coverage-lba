#!/usr/bin/env python3
"""Regenerate every publication figure into ``figures/`` (vector PDF + raster PNG).

Four figure families are produced, each read directly from a result document --
nothing in any panel is typed in by hand:

===========================  ==============================================  =========
Family                       Source document                                 Files
===========================  ==============================================  =========
Step 1 engine validation     ``results/step1_guidance_validation.json``       fig1-fig8
                             ``data/step1_trials.json``
Revision / order sweep       ``results/step2_revision.json``                  fig9-fig15
                             ``data/t4_order_sweep.json``
Step 2 scheduler campaign    ``data/step2_trials.json``                       step2_*
Consolidated paper figures   ``results/step2_revision.json``                  paper_fig1-5
                             ``results/corpus_cap_ablation.json``
===========================  ==============================================  =========

The consolidated paper figures are the ones that appear in the submission; the
others are the extended-manuscript versions that show the same data at lower
density.  Both are regenerated so a reviewer can check that consolidation did not
change any number.

Because the source documents must exist first, this script will run
``experiments/analyze_results.py`` for you if ``results/`` has not been populated.

Usage
-----
    python plots/generate_all_figures.py
    python plots/generate_all_figures.py --only paper
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

AE_ROOT = Path(__file__).resolve().parent.parent
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))

import ae_paths  # noqa: E402

FIGURES = ae_paths.FIGURES
RESULTS = ae_paths.RESULTS

#: Documents each family needs before it can be drawn.
REQUIRED: Dict[str, List[str]] = {
    "step1": ["step1_guidance_validation.json"],
    "revision": ["step2_revision.json"],
    "step2": [],  # drawn straight from data/step2_trials.json
    "paper": ["step2_revision.json", "corpus_cap_ablation.json"],
}


def _ensure_results() -> None:
    """Populate ``results/`` by running the analysis driver if it looks empty."""
    needed = {n for names in REQUIRED.values() for n in names}
    missing = [n for n in needed if not (RESULTS / n).exists()]
    if not missing:
        return
    print(f"[figures] results/ is missing {missing}; running analyze_results.py first",
          flush=True)
    rc = subprocess.run(
        [sys.executable, str(AE_ROOT / "experiments" / "analyze_results.py")],
        cwd=str(AE_ROOT),
    ).returncode
    if rc != 0:
        raise SystemExit(f"analyze_results.py failed with exit code {rc}")


def _draw_step1() -> None:
    """Figures 1-8: engine validation, bug discovery, overhead, collisions."""
    sys.path.insert(0, str(AE_ROOT / "plots"))
    import _figures_step1 as f1
    f1.main()


def _draw_revision() -> None:
    """Figures 9-15: information loss, lattice, order sweep, paired outcomes."""
    sys.path.insert(0, str(AE_ROOT / "plots"))
    import _figures_revision as f2
    f2.main()


def _draw_step2() -> None:
    """step2_* figures: survival, coverage growth, energy, dilution, depth."""
    sys.path.insert(0, str(AE_ROOT / "experiments"))
    import _analysis_step2 as an
    raw = json.loads((ae_paths.DATA / "step2_trials.json").read_text())
    trials = raw["trials"]
    budget = int(raw["max_execs"])
    targets = raw["targets"]
    configs = [c["name"] for c in raw["configs"]]
    FIGURES.mkdir(parents=True, exist_ok=True)
    an._fig_survival(targets, configs, trials, budget)
    an._fig_dilution(targets, configs, trials)
    an._fig_energy(targets, configs, trials)
    an._fig_coverage(targets, configs, trials)
    an._fig_depth(targets, trials)


def _draw_paper() -> None:
    """The five consolidated figures used in the submission itself."""
    sys.path.insert(0, str(AE_ROOT / "plots"))
    import _figures_paper as fp
    fp.fig_factorisation()
    fp.fig_infoloss()
    fp.fig_discrimination()
    fp.fig_ordersweep()
    fp.fig_capablation()


FAMILIES: Dict[str, Callable[[], None]] = {
    "step1": _draw_step1,
    "revision": _draw_revision,
    "step2": _draw_step2,
    "paper": _draw_paper,
}


def main() -> int:
    """Draw the requested figure families and write a figure manifest."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=str, default="",
                    help=f"comma-separated subset of {sorted(FAMILIES)}")
    args = ap.parse_args()

    wanted = [f for f in args.only.split(",") if f] or list(FAMILIES)
    unknown = [f for f in wanted if f not in FAMILIES]
    if unknown:
        print(f"unknown figure family/families: {unknown}; "
              f"choose from {sorted(FAMILIES)}", file=sys.stderr)
        return 2

    _ensure_results()
    FIGURES.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(FIGURES)) for p in FIGURES.rglob("*") if p.is_file()}

    t0 = time.time()
    status: Dict[str, Any] = {}
    for fam in wanted:
        t1 = time.time()
        print(f"[figures] drawing family '{fam}' ...", flush=True)
        try:
            FAMILIES[fam]()
            status[fam] = {"ok": True, "seconds": round(time.time() - t1, 2)}
            print(f"[figures]   '{fam}' done in {status[fam]['seconds']}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            status[fam] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[figures]   '{fam}' FAILED: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)

    after = sorted(str(p.relative_to(FIGURES)) for p in FIGURES.rglob("*")
                   if p.is_file())
    pdfs = [n for n in after if n.endswith(".pdf")]
    pngs = [n for n in after if n.endswith(".png")]
    manifest = {
        "schema": "ae_figure_manifest/1.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families_requested": wanted,
        "family_status": status,
        "n_pdf": len(pdfs),
        "n_png": len(pngs),
        "new_this_run": sorted(set(after) - before),
        "files": after,
        "wall_seconds": round(time.time() - t0, 2),
    }
    (RESULTS / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))

    ok = all(s.get("ok") for s in status.values())
    print(f"[figures] {len(pdfs)} PDF + {len(pngs)} PNG in figures/  "
          f"({manifest['wall_seconds']}s)", flush=True)
    print(f"[figures] manifest -> results/figure_manifest.json", flush=True)
    print(f"[figures] {'OK' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
