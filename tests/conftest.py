"""pytest bootstrap for the artifact-evaluation test suite.

Puts the AE root (and therefore ``src/`` and ``targets/`` via :mod:`ae_paths`) on
``sys.path`` before any test module is imported, and makes sure the native shared
libraries have been compiled -- the invariant and fidelity suites both compare the
Python target models against the native C ground truth, so a missing ``.so`` would
otherwise surface as a confusing import-time error.
"""
from __future__ import annotations

import sys
from pathlib import Path

AE_ROOT = Path(__file__).resolve().parent.parent
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))
if str(AE_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(AE_ROOT / "tests"))

import ae_paths  # noqa: E402,F401  (registers src/ and targets/ on sys.path)


def pytest_configure(config) -> None:
    """Compile the native benchmark libraries once, before collection runs."""
    from test_targets import build_native_lib
    from trigram_targets import build_trigram_lib

    build_native_lib()
    build_trigram_lib()
