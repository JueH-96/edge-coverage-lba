"""Single source of truth for every path inside the artifact-evaluation package.

Importing this module has two effects:

1. It defines the canonical directory constants (``SRC``, ``NATIVE``, ``TARGETS``,
   ``DATA``, ``RESULTS``, ``FIGURES``, ...), all resolved relative to this file, so
   the package can be unpacked anywhere.
2. It registers ``src/`` and ``targets/`` on ``sys.path``, so the ported research
   modules can keep importing each other by plain module name
   (``import guidance_engine``, ``from test_targets import get_targets``) exactly as
   they did in the original session tree.  No relative-import surgery was applied to
   the research code, which keeps the ported files byte-comparable to the originals.

There are no absolute paths anywhere in this package.
"""
from __future__ import annotations

import sys
from pathlib import Path

#: Root of the artifact-evaluation package (the directory holding this file).
AE_ROOT: Path = Path(__file__).resolve().parent

SRC: Path = AE_ROOT / "src"
NATIVE: Path = AE_ROOT / "native"
TARGETS: Path = AE_ROOT / "targets"
TESTS: Path = AE_ROOT / "tests"
EXPERIMENTS: Path = AE_ROOT / "experiments"
PLOTS: Path = AE_ROOT / "plots"
DATA: Path = AE_ROOT / "data"
PRECOMPUTED: Path = DATA / "precomputed"
RESULTS: Path = AE_ROOT / "results"
FIGURES: Path = AE_ROOT / "figures"

#: Directories that must exist before anything writes into them.
_OUTPUT_DIRS = (RESULTS, FIGURES)

#: Directories placed on ``sys.path`` so the ported modules resolve each other.
_IMPORT_DIRS = (AE_ROOT, SRC, TARGETS)


def _register() -> None:
    """Put the package's import directories at the front of ``sys.path``."""
    for d in reversed(_IMPORT_DIRS):
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


def ensure_output_dirs() -> None:
    """Create ``results/`` and ``figures/`` if they do not exist yet."""
    for d in _OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def describe() -> dict:
    """Return a JSON-serialisable snapshot of the resolved layout.

    Returns
    -------
    dict
        Mapping of logical name to path *relative to the AE root*, plus the
        absolute root itself.  Used by the runner scripts for provenance.
    """
    return {
        "ae_root": str(AE_ROOT),
        "layout": {
            name: str(path.relative_to(AE_ROOT))
            for name, path in (
                ("src", SRC), ("native", NATIVE), ("targets", TARGETS),
                ("tests", TESTS), ("experiments", EXPERIMENTS), ("plots", PLOTS),
                ("data", DATA), ("precomputed", PRECOMPUTED),
                ("results", RESULTS), ("figures", FIGURES),
            )
        },
    }


_register()
ensure_output_dirs()
