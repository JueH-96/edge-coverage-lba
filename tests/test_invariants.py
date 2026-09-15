"""The complete invariant suite: 75 tests over the theory and the implementation.

This module is a thin aggregator.  The tests themselves live in two non-collected
modules that were ported verbatim from the research session:

* ``_invariants_step1.py`` -- 22 tests covering the guidance engine itself: the
  Refinement Theorem (D1 partitions refine AFL edge partitions), the O(1) rolling
  prefix-hash against an O(N) reference implementation, value-range bucketing
  semantics, the Miller-Madow entropy estimator, Jeffreys-smoothed surprisal, and
  the AFL-bigram characterisation that underpins the paper's main theoretical claim.
* ``_invariants_step2.py`` -- 53 tests (38 functions, some parametrised over the
  three trigram targets) covering the energy schedules, the trigram blind-spot
  targets, state transitions, and the corpus-dilution metrics (sterile fraction,
  energy Gini).

Importing the test functions here rather than duplicating them keeps a single
definition of every invariant.  ``pytest`` collects imported test functions, so the
count reported for this file is the full 75.

Run with::

    python -m pytest tests/test_invariants.py -v
"""
from __future__ import annotations

# The star-imports below are the point of this module: they re-export every
# ``test_*`` function so pytest collects the whole suite from one file.
from _invariants_step1 import *  # noqa: F401,F403
from _invariants_step2 import *  # noqa: F401,F403
