#!/usr/bin/env python3
"""Compile the two native C benchmark libraries into ``native/``.

The C sources are the *ground truth* for every benchmark target: the Python models
used by the experiments are only admissible because they agree with these libraries
exactly (see ``tests/test_fidelity.py``).  Both are built with the same flags used
for the results reported in the paper::

    gcc -O2 -fPIC -shared -o native/libtarget.so native/target_lib.c
    gcc -O2 -fPIC -shared -o native/libtrigram_target.so native/trigram_target_lib.c

The build is idempotent: a library newer than its source is left alone unless
``--force`` is given.

Usage
-----
    python experiments/build_native.py [--force]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

AE_ROOT = Path(__file__).resolve().parent.parent
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))

import ae_paths  # noqa: E402

from test_targets import build_native_lib  # noqa: E402
from trigram_targets import build_trigram_lib  # noqa: E402


def main() -> int:
    """Build both shared libraries and report their locations and sizes."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the .so is newer than its source")
    args = ap.parse_args()

    try:
        cc = subprocess.run(["gcc", "--version"], capture_output=True, text=True,
                            check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        print(f"FATAL: a C compiler is required but gcc could not be run: {exc}",
              file=sys.stderr)
        return 1
    print(f"[build] compiler: {cc}", flush=True)

    built = {}
    for label, fn in (("libtarget.so", build_native_lib),
                      ("libtrigram_target.so", build_trigram_lib)):
        path = fn(force=args.force)
        size = path.stat().st_size
        built[label] = {"path": str(path.relative_to(AE_ROOT)), "size_bytes": size}
        print(f"[build] {label:24s} -> {path.relative_to(AE_ROOT)} "
              f"({size / 1024:.0f} kB)", flush=True)

    ae_paths.ensure_output_dirs()
    (ae_paths.RESULTS / "build_native.json").write_text(
        json.dumps({"compiler": cc, "libraries": built}, indent=2)
    )
    print("[build] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
