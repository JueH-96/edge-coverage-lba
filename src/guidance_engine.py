"""
3D Multidimensional Guidance Engine for coverage-guided fuzzing of C/C++ native libraries.

Step 1 core module: theoretical model + modular tracking engine.

Theoretical model
-----------------
Let an execution of a target on input ``x`` be observed through a set of
instrumentation probes. Each *dimension* ``i`` of the guidance engine defines an
abstraction function

    alpha_i : Exec -> 2^{U_i}

mapping an execution to a finite set of *coverage elements* drawn from a universe
``U_i``. Two executions are equivalent under dimension ``i`` iff
``alpha_i(x) == alpha_i(x')``; each ``alpha_i`` therefore induces a partition
``Pi_i`` of the input space, and a coverage-guided fuzzer retains an input exactly
when it lands in a previously unvisited cell (or, with hit-count bucketing, a
previously unvisited (cell, count-class) pair).

The four dimensions implemented here are

===  ================================  ===============================================
Dim  Tracker                           Sensitivity captured
===  ================================  ===============================================
D0   :class:`AFLEdgeTracker`           control-flow transitions (AFL-style baseline)
D1   :class:`CallingContextTracker`    *where from* - bounded N-depth calling context
D2   :class:`ValueRangeTracker`        *with what values* - operand magnitude / cmp distance
D3   :class:`StateMachineTracker`      *in what order* - implicit API state transitions
===  ================================  ===============================================

**Refinement Theorem (design invariant).** ``AFLEdgeTracker`` keys elements by the
edge index ``e = loc(prev) ^ loc(cur)``; ``CallingContextTracker`` keys elements by
the pair ``(ctx, e)`` over the *same* edge stream. Let ``pi(ctx, e) = e`` be the
projection that discards the context component. Then

    alpha_0 = pi o alpha_1                       (1)

and consequently ``alpha_1(x) == alpha_1(x')  =>  alpha_0(x) == alpha_0(x')``, i.e.
``Pi_1`` is a **refinement** of ``Pi_0``. D1 therefore never loses information
relative to AFL edge coverage, and strictly gains whenever two executions traverse
identical edges from different calling contexts. Equation (1) is asserted as a unit
test (``test_refinement_theorem``) over exact, collision-free element sets rather
than assumed.

Refinement is *not free*. For every dimension the engine also exposes the cost side
of the ledger, which the Step-1 validation reports explicitly:

* state-space cardinality ``|alpha_i(corpus)|`` (dimensionality inflation, which
  dilutes the corpus-admission signal and grows the queue),
* hash-collision rate at a bounded map size ``2^b`` (birthday bound
  ``E[collisions] ~ k^2 / 2^(b+1)`` for ``k`` distinct elements), measured against
  shadow sets of *exact* elements,
* per-execution wall-clock overhead relative to the D0 baseline.

Novelty scoring is information-theoretic rather than ad hoc:

* D2 reports Shannon entropy of each value node's bucket histogram using both the
  plug-in (maximum-likelihood) estimator and the **Miller-Madow** bias correction
  ``H_MM = H_plugin + (K - 1) / (2 N ln 2)`` bits.
* D3 weights a transition ``t`` by its surprisal ``-log2 p_hat(t)`` under an
  add-alpha (Jeffreys, ``alpha = 0.5``) smoothed categorical model with a reserved
  escape symbol, so never-seen transitions receive maximal, finite, well-defined
  weight and the weight decreases monotonically with observation count.

Prior-art anchors
-----------------
D0  AFL / AFL++ ``has_new_bits`` and ``count_class_lookup8``.
D1  Probabilistic Calling Context (Bond & McKinley, OOPSLA 2007); context-sensitive
    coverage-guided fuzzing (Wang et al., "Be Sensitive and Collaborative", RAID 2019);
    AFL++ ``AFL_LLVM_CTX``.
D2  laf-intel comparison splitting; RedQueen (Aschermann et al., NDSS 2019);
    AFL++ CmpLog.
D3  AFLNet (Pham et al., ICST 2020); StateAFL (Natella, EMSE 2022); IJON (S&P 2020).

Notes
-----
This module is pure Python by design: Step 1 validates the *algorithms and their
information content*, and a portable reference implementation makes the invariants
testable. The instrumentation-probe API (:class:`MultiDimGuidanceEngine.bb`,
``enter``/``leave``, ``cmp_``, ``value``, ``api``) is deliberately shaped like what a
compiler pass or DynamoRIO/QBDI client would emit, so the same engine can be driven
by native instrumentation in later steps.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "AFLEdgeTracker",
    "CallingContextTracker",
    "ValueRangeTracker",
    "StateMachineTracker",
    "NGramTracker",
    "MultiDimGuidanceEngine",
    "Feedback",
    "shannon_entropy_bits",
    "miller_madow_entropy_bits",
    "afl_count_class",
    "DIMENSIONS",
]

MASK64 = (1 << 64) - 1
_LN2 = math.log(2.0)

#: Canonical dimension identifiers. ``DN`` is the AFL++ ``NGRAM-k`` baseline: it is
#: not part of the proposed engine, it is a *competing* published abstraction that
#: the order sweep needs as a reference point, so it is kept separate from the
#: four dimensions of the engine proper.
DIMENSIONS = ("D0", "D1", "D2", "D3", "DN")


# --------------------------------------------------------------------------- #
# Hashing / bucketing primitives
# --------------------------------------------------------------------------- #
def _mix64(x: int) -> int:
    """SplitMix64 finalizer - strong avalanche, used for all id derivation.

    Parameters
    ----------
    x : int
        Arbitrary integer (reduced mod 2**64).

    Returns
    -------
    int
        64-bit mixed value.
    """
    x &= MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def _build_afl_count_class_table() -> np.ndarray:
    """AFL's ``count_class_lookup8``: 8 saturating hit-count classes.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of length 256 mapping a raw hit count to its class bit.
    """
    t = np.zeros(256, dtype=np.uint8)
    t[0] = 0
    t[1] = 1
    t[2] = 2
    t[3] = 4
    t[4:8] = 8
    t[8:16] = 16
    t[16:32] = 32
    t[32:128] = 64
    t[128:256] = 128
    return t


_AFL_COUNT_CLASS = _build_afl_count_class_table()


def afl_count_class(count: int) -> int:
    """Map a raw hit count to AFL's saturating hit-count class bit.

    Parameters
    ----------
    count : int
        Raw execution count for one edge (values >= 128 saturate).

    Returns
    -------
    int
        One of ``{0, 1, 2, 4, 8, 16, 32, 64, 128}``.
    """
    return int(_AFL_COUNT_CLASS[min(int(count), 255)])


def shannon_entropy_bits(counts: Iterable[int]) -> float:
    """Plug-in (maximum-likelihood) Shannon entropy in bits.

    Parameters
    ----------
    counts : iterable of int
        Observation counts per category; zeros are ignored.

    Returns
    -------
    float
        ``-sum p log2 p`` with ``p = c / N``. Returns 0.0 for an empty sample.
    """
    cs = [int(c) for c in counts if c > 0]
    n = sum(cs)
    if n <= 0:
        return 0.0
    h = 0.0
    for c in cs:
        p = c / n
        h -= p * math.log2(p)
    return h


def miller_madow_entropy_bits(counts: Iterable[int]) -> float:
    """Miller-Madow bias-corrected Shannon entropy in bits.

    The plug-in estimator is negatively biased by approximately ``(K - 1) / (2 N)``
    nats, where ``K`` is the number of observed categories; this adds the correction
    back and converts to bits.

    Parameters
    ----------
    counts : iterable of int
        Observation counts per category.

    Returns
    -------
    float
        ``H_plugin + (K - 1) / (2 N ln 2)``.
    """
    cs = [int(c) for c in counts if c > 0]
    n = sum(cs)
    if n <= 0:
        return 0.0
    k = len(cs)
    return shannon_entropy_bits(cs) + (k - 1) / (2.0 * n * _LN2)


def signed_log2_bucket(value: int) -> int:
    """Discretize a signed integer into a symmetric log2-magnitude bucket.

    Bucket 64 is exactly zero; buckets 65..128 are positive magnitudes with bit
    length 1..64; buckets 63..0 are the negative mirror. This mirrors AFL++'s
    magnitude bucketing for loop counters, lengths and pointer offsets, keeping the
    universe small (129 buckets) while preserving order-of-magnitude information.

    Parameters
    ----------
    value : int
        Observed value.

    Returns
    -------
    int
        Bucket index in ``[0, 128]``.
    """
    v = int(value)
    if v == 0:
        return 64
    bits = min(64, abs(v).bit_length())
    return 64 + bits if v > 0 else 64 - bits


def distance_bucket(a: int, b: int) -> int:
    """Discretize the *distance* between two comparison operands.

    Bucket 0 means the operands were equal (the comparison was satisfied); bucket
    ``1 + floor(log2 |a - b|)`` otherwise. Unlike edge coverage, this preserves a
    monotone gradient toward satisfying an equality comparison, which is the signal
    laf-intel and RedQueen exploit.

    Parameters
    ----------
    a, b : int
        Comparison operands.

    Returns
    -------
    int
        Bucket index in ``[0, 64]``.
    """
    d = abs(int(a) - int(b))
    if d == 0:
        return 0
    return 1 + min(63, d.bit_length() - 1)


def common_byte_prefix(a: int, b: int, width: int, msb_first: bool = True) -> int:
    """Number of leading equal bytes of two operands at a given width.

    Parameters
    ----------
    a, b : int
        Comparison operands (interpreted as unsigned ``width``-byte integers).
    width : int
        Operand width in bytes (1, 2, 4 or 8).
    msb_first : bool, optional
        Count from the most significant byte (big-endian view) when True, else from
        the least significant byte. Both views are tracked by
        :class:`ValueRangeTracker` because real magic-value fields occur in both
        byte orders.

    Returns
    -------
    int
        Count of matching leading bytes in ``[0, width]``.
    """
    mask = (1 << (8 * width)) - 1
    ua = int(a) & mask
    ub = int(b) & mask
    order = range(width - 1, -1, -1) if msb_first else range(width)
    n = 0
    for i in order:
        if ((ua >> (8 * i)) & 0xFF) != ((ub >> (8 * i)) & 0xFF):
            break
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Shared AFL-style bounded bitmap
# --------------------------------------------------------------------------- #
class _AFLBitmap:
    """AFL ``trace_bits`` / ``virgin_bits`` pair with saturating hit-count classes.

    The per-execution trace is a sparse dict (``index -> raw count``) so that
    resetting costs ``O(touched)`` rather than ``O(map_size)``; this matters because
    a Python reference implementation is otherwise dominated by map clearing.

    Parameters
    ----------
    map_bits : int
        Log2 of the map size (AFL's default is 16, i.e. a 64 KiB map).
    """

    __slots__ = ("map_bits", "map_size", "mask", "virgin", "trace")

    def __init__(self, map_bits: int = 16) -> None:
        self.map_bits = int(map_bits)
        self.map_size = 1 << self.map_bits
        self.mask = self.map_size - 1
        self.virgin = np.full(self.map_size, 0xFF, dtype=np.uint8)
        self.trace: Dict[int, int] = {}

    def reset(self) -> None:
        """Clear the per-execution trace."""
        self.trace.clear()

    def hit(self, index: int) -> None:
        """Record one hit at ``index`` with saturating count at 255."""
        idx = index & self.mask
        c = self.trace.get(idx, 0)
        if c < 255:
            self.trace[idx] = c + 1

    def has_new_bits(self, commit: bool = True) -> int:
        """AFL ``has_new_bits`` semantics.

        Parameters
        ----------
        commit : bool, optional
            When True, fold the current trace into ``virgin_bits`` (i.e. actually
            claim the new coverage). When False, the check is non-destructive.

        Returns
        -------
        int
            ``0`` if nothing new, ``1`` if only a new hit-count class was seen on an
            already-known element, ``2`` if a previously unreachable element was hit.
        """
        virgin = self.virgin
        ret = 0
        table = _AFL_COUNT_CLASS
        for idx, raw in self.trace.items():
            cls = int(table[raw if raw < 256 else 255])
            if cls == 0:
                continue
            v = int(virgin[idx])
            if cls & v:
                if v == 0xFF:
                    ret = 2
                elif ret < 1:
                    ret = 1
                if commit:
                    virgin[idx] = v & (~cls & 0xFF)
        return ret

    @property
    def covered(self) -> int:
        """Number of map slots that have been hit at least once."""
        return int(np.count_nonzero(self.virgin != 0xFF))


# --------------------------------------------------------------------------- #
# D0 - AFL-style edge coverage (control baseline)
# --------------------------------------------------------------------------- #
class AFLEdgeTracker:
    """Standard 2D AFL-style edge coverage, used as the experimental control.

    Faithful to AFL's instrumentation contract:

    * every basic block gets a compile-time-random location ``loc(bb)`` in
      ``[0, map_size)``,
    * the bitmap index is ``prev_loc ^ cur_loc``,
    * ``prev_loc`` is updated to ``cur_loc >> 1`` so that ``A -> B`` and ``B -> A``
      map to different slots,
    * hit counts are folded into 8 saturating classes before novelty checking.

    A shadow set of *exact* ``(prev_bb, cur_bb)`` pairs is maintained so the
    bitmap's collision rate can be measured rather than assumed.

    Parameters
    ----------
    map_bits : int, optional
        Log2 of the bitmap size, default 16 (AFL's 64 KiB default).
    track_exact : bool, optional
        Maintain the exact-element shadow set (needed for collision accounting and
        for the refinement-theorem test). Default True.

    Attributes
    ----------
    bitmap : _AFLBitmap
        Bounded map with AFL ``virgin_bits`` semantics.
    exact_elements : set of tuple
        Global set of exact ``(prev_bb, cur_bb)`` edges observed.
    """

    dimension = "D0"

    def __init__(self, map_bits: int = 16, track_exact: bool = True) -> None:
        self.bitmap = _AFLBitmap(map_bits)
        self.track_exact = bool(track_exact)
        self.exact_elements: Set[Tuple[Any, Any]] = set()
        #: exact per-execution edge -> raw hit count (enables a hit-count-aware
        #: signature, which is what AFL actually discriminates on)
        self.exact_trace: Dict[Tuple[Any, Any], int] = {}
        self._loc_cache: Dict[Any, int] = {}
        self._prev_loc = 0
        self._prev_bb: Any = None
        self.n_probes = 0

    # -- instrumentation ---------------------------------------------------- #
    def loc(self, bb_id: Hashable) -> int:
        """Deterministic compile-time-style location id for a basic block."""
        v = self._loc_cache.get(bb_id)
        if v is None:
            v = _mix64(hash(bb_id) * 0x9E3779B97F4A7C15) & self.bitmap.mask
            self._loc_cache[bb_id] = v
        return v

    def bb(self, bb_id: Hashable) -> None:
        """Instrumentation probe: basic block ``bb_id`` was entered."""
        cur = self.loc(bb_id)
        self.bitmap.hit(self._prev_loc ^ cur)
        if self.track_exact:
            k = (self._prev_bb, bb_id)
            t = self.exact_trace
            c = t.get(k, 0)
            if c < 255:
                t[k] = c + 1
        self._prev_loc = cur >> 1
        self._prev_bb = bb_id
        self.n_probes += 1

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        """Begin a new execution: clear the per-execution trace and chain state."""
        self.bitmap.reset()
        self.exact_trace = {}
        self._prev_loc = 0
        self._prev_bb = None

    def trace_signature(self) -> frozenset:
        """Exact per-execution edge set (used for partition analysis)."""
        return frozenset(self.exact_trace.keys())

    def trace_signature_counted(self) -> frozenset:
        """Exact per-execution ``(edge, AFL hit-count class)`` set.

        This is the *strongest* form of the AFL baseline signature: two executions
        with equal counted signatures are indistinguishable to AFL even with its
        saturating hit-count classes enabled. Used to certify that the baseline is
        genuinely blind to a behavioural difference rather than merely slower.
        """
        return frozenset(
            (k, afl_count_class(v)) for k, v in self.exact_trace.items()
        )

    def evaluate(self, commit: bool = True) -> Dict[str, Any]:
        """Check the current execution for novelty.

        Parameters
        ----------
        commit : bool, optional
            Claim the coverage into the global virgin map when True.

        Returns
        -------
        dict
            ``afl_status`` (0/1/2), ``new_elements`` (exact, collision-free count),
            and ``is_new``.
        """
        status = self.bitmap.has_new_bits(commit=commit)
        new_exact = 0
        if self.track_exact:
            fresh = self.exact_trace.keys() - self.exact_elements
            new_exact = len(fresh)
            if commit and fresh:
                self.exact_elements |= fresh
        return {"afl_status": status, "new_elements": new_exact, "is_new": status > 0}

    # -- accounting --------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        """Coverage and collision statistics for this dimension."""
        exact = len(self.exact_elements)
        mapped = self.bitmap.covered
        return {
            "dimension": self.dimension,
            "map_bits": self.bitmap.map_bits,
            "exact_elements": exact,
            "mapped_slots_covered": mapped,
            "collisions": max(0, exact - mapped),
            "collision_rate": (exact - mapped) / exact if exact else 0.0,
            "expected_collisions_birthday": (exact * exact) / (2 ** (self.bitmap.map_bits + 1))
            if exact
            else 0.0,
            "probe_calls": self.n_probes,
        }


# --------------------------------------------------------------------------- #
# DN - AFL++ NGRAM-k baseline (competing published abstraction, not ours)
# --------------------------------------------------------------------------- #
class NGramTracker:
    """AFL++ ``NGRAM-k`` coverage: a sliding window over the executed block stream.

    AFL++'s ``AFL_LLVM_INSTRUMENT=NGRAM-k`` replaces AFL's single ``prev_loc``
    with a shift register of the ``k-1`` previous locations and indexes the map
    with their XOR against the current location::

        idx = cur_loc ^ prev[0] ^ prev[1] ^ ... ^ prev[k-2]
        prev = (cur_loc >> 1) : prev[0 .. k-3]

    With ``k = 2`` this degenerates to exactly AFL's edge index, which is why the
    order sweep uses ``DN(2)`` as an internal consistency check against
    :class:`AFLEdgeTracker`.

    The distinction from :class:`CallingContextTracker` matters and is easy to
    miss: NGRAM-k windows the *executed block sequence*, whereas D1 windows the
    *call stack*. On a straight-line block sequence the two coincide; on a program
    that returns from a call they do not, because the stack pops while the block
    stream does not. Both are reported so the order sweep separates "order-k
    information" from "the particular way a fuzzer obtains it".

    The exact shadow element is the ordered tuple of the last ``k`` block ids,
    which is precisely the ``k``-gram of the executed sequence - so the tracker's
    exact element set is literally the ``k``-gram spectrum ``G_k`` used in the
    theory section.

    Parameters
    ----------
    k : int, optional
        Window length (``k = 2`` is plain edge coverage), default 3.
    map_bits : int, optional
        Log2 bitmap size, default 16.
    track_exact : bool, optional
        Maintain the exact ``k``-gram shadow set, default True.
    """

    dimension = "DN"

    def __init__(self, k: int = 3, map_bits: int = 16, track_exact: bool = True) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.bitmap = _AFLBitmap(map_bits)
        self.track_exact = bool(track_exact)
        self.exact_elements: Set[Tuple[Any, ...]] = set()
        self.exact_trace: Dict[Tuple[Any, ...], int] = {}
        self._loc_cache: Dict[Any, int] = {}
        self._prev_loc: List[int] = [0] * (self.k - 1)
        self._prev_bb: List[Any] = [None] * (self.k - 1)
        self.n_probes = 0

    def loc(self, bb_id: Hashable) -> int:
        """Deterministic location id for a basic block."""
        v = self._loc_cache.get(bb_id)
        if v is None:
            v = _mix64(hash(bb_id) * 0x9E3779B97F4A7C15) & self.bitmap.mask
            self._loc_cache[bb_id] = v
        return v

    def bb(self, bb_id: Hashable) -> None:
        """Instrumentation probe: basic block entered."""
        cur = self.loc(bb_id)
        idx = cur
        for v in self._prev_loc:
            idx ^= v
        self.bitmap.hit(idx)
        if self.track_exact:
            key = tuple(self._prev_bb) + (bb_id,)
            t = self.exact_trace
            c = t.get(key, 0)
            if c < 255:
                t[key] = c + 1
        if self.k > 1:
            self._prev_loc = [cur >> 1] + self._prev_loc[: self.k - 2]
            self._prev_bb = [bb_id] + self._prev_bb[: self.k - 2]
        self.n_probes += 1

    def reset(self) -> None:
        """Begin a new execution."""
        self.bitmap.reset()
        self.exact_trace = {}
        self._prev_loc = [0] * (self.k - 1)
        self._prev_bb = [None] * (self.k - 1)

    def trace_signature(self) -> frozenset:
        """Exact per-execution ``k``-gram set."""
        return frozenset(self.exact_trace.keys())

    def trace_signature_counted(self) -> frozenset:
        """Exact per-execution ``(k``-gram, AFL hit-count class) set."""
        return frozenset((k, afl_count_class(v)) for k, v in self.exact_trace.items())

    def evaluate(self, commit: bool = True) -> Dict[str, Any]:
        """Check the current execution for ``k``-gram novelty."""
        status = self.bitmap.has_new_bits(commit=commit)
        new_exact = 0
        if self.track_exact:
            fresh = self.exact_trace.keys() - self.exact_elements
            new_exact = len(fresh)
            if commit and fresh:
                self.exact_elements |= fresh
        return {"afl_status": status, "new_elements": new_exact, "is_new": status > 0}

    def stats(self) -> Dict[str, Any]:
        """Coverage and collision statistics for the NGRAM baseline."""
        exact = len(self.exact_elements)
        mapped = self.bitmap.covered
        return {
            "dimension": self.dimension,
            "ngram_k": self.k,
            "map_bits": self.bitmap.map_bits,
            "exact_elements": exact,
            "mapped_slots_covered": mapped,
            "collisions": max(0, exact - mapped),
            "collision_rate": (exact - mapped) / exact if exact else 0.0,
            "expected_collisions_birthday": (exact * exact)
            / (2 ** (self.bitmap.map_bits + 1))
            if exact
            else 0.0,
            "probe_calls": self.n_probes,
        }


# --------------------------------------------------------------------------- #
# D1 - Calling-context sensitivity
# --------------------------------------------------------------------------- #
class CallingContextTracker:
    """Bounded N-depth calling-context sensitive edge coverage.

    Context encoding
    ----------------
    A stack of *prefix hashes* is maintained alongside the call-site stack:

        H_0 = 0,   H_k = H_{k-1} * P + c_k   (mod 2**64)

    The hash of the **top N frames only** is then recovered in ``O(1)`` as

        W = H_m - H_{m-n} * P^n   (mod 2**64),   n = min(N, m)

    so ``enter``/``leave`` are constant-time regardless of ``N`` (a naive
    implementation re-hashes the window on every call, costing ``O(N)``). The window
    depth ``n`` is mixed into the final id so that stacks of different depth cannot
    alias through zero-valued call-site ids. This is a bounded-depth variant of
    Probabilistic Calling Context (Bond & McKinley, OOPSLA 2007); bounding the depth
    is what keeps the context space finite in the presence of deep call chains.

    Recursion folding
    -----------------
    Unbounded recursion would otherwise generate a fresh context per recursion
    depth and explode the state space. Consecutive identical call sites beyond
    ``recursion_cap`` are *folded*: they do not contribute a new effective frame.
    The resulting effective-stack depth for a self-recursive call site is therefore
    bounded by ``recursion_cap``, which is asserted by a unit test.

    Coverage element
    ----------------
    ``(ctx, prev_loc ^ cur_loc)``, mapped into the bitmap as
    ``(prev_loc ^ cur_loc) ^ (ctx & mask)`` following AFL++'s ``AFL_LLVM_CTX``. The
    exact shadow element is ``(exact_context_tuple, prev_bb, cur_bb)``, whose
    projection onto ``(prev_bb, cur_bb)`` is exactly :class:`AFLEdgeTracker`'s
    element - the basis of the Refinement Theorem.

    Parameters
    ----------
    depth : int, optional
        Context window ``N``, default 4.
    map_bits : int, optional
        Log2 bitmap size, default 16.
    recursion_cap : int, optional
        Maximum consecutive identical call-site frames that contribute, default 2.
    track_exact : bool, optional
        Maintain exact shadow element sets, default True.
    """

    dimension = "D1"

    def __init__(
        self,
        depth: int = 4,
        map_bits: int = 16,
        recursion_cap: int = 2,
        track_exact: bool = True,
        prime: int = 0x100000001B3,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if recursion_cap < 1:
            raise ValueError("recursion_cap must be >= 1")
        self.depth = int(depth)
        self.recursion_cap = int(recursion_cap)
        self.P = int(prime) & MASK64
        self.bitmap = _AFLBitmap(map_bits)
        self.track_exact = bool(track_exact)

        self._pow = [1]
        self._extend_pow(self.depth + 1)

        # effective (post-folding) call-site stack + parallel prefix hashes
        self._eff: List[int] = []
        self._prefix: List[int] = [0]
        # bookkeeping per physical frame: (contributed, previous_run_length)
        self._frames: List[Tuple[bool, int]] = []
        self._run = 0

        self._loc_cache: Dict[Any, int] = {}
        self._prev_loc = 0
        self._prev_bb: Any = None

        self.contexts: Set[int] = set()
        self.exact_contexts: Set[Tuple[int, ...]] = set()
        self.exact_elements: Set[Tuple[Tuple[int, ...], Any, Any]] = set()
        self.exact_trace: Set[Tuple[Tuple[int, ...], Any, Any]] = set()
        self.context_trace: Set[int] = set()
        self.max_effective_depth = 0
        self.n_probes = 0

    # -- internals ---------------------------------------------------------- #
    def _extend_pow(self, upto: int) -> None:
        while len(self._pow) <= upto:
            self._pow.append((self._pow[-1] * self.P) & MASK64)

    def loc(self, bb_id: Hashable) -> int:
        """Deterministic location id for a basic block."""
        v = self._loc_cache.get(bb_id)
        if v is None:
            v = _mix64(hash(bb_id) * 0x9E3779B97F4A7C15) & self.bitmap.mask
            self._loc_cache[bb_id] = v
        return v

    # -- instrumentation ---------------------------------------------------- #
    def enter(self, callsite_id: int) -> None:
        """Instrumentation probe: a call is being made from ``callsite_id``.

        ``O(1)``: one multiply-add for the prefix hash plus constant bookkeeping.
        """
        cs = int(callsite_id) & MASK64
        top = self._eff[-1] if self._eff else None
        prev_run = self._run
        if top == cs and self._run >= self.recursion_cap:
            contributed = False  # fold this recursion frame
        else:
            contributed = True
            self._run = (prev_run + 1) if top == cs else 1
            self._eff.append(cs)
            self._prefix.append(((self._prefix[-1] * self.P) + cs) & MASK64)
            if len(self._eff) > self.max_effective_depth:
                self.max_effective_depth = len(self._eff)
        self._frames.append((contributed, prev_run))

    def leave(self) -> None:
        """Instrumentation probe: return from the most recent call."""
        if not self._frames:
            raise RuntimeError("leave() without matching enter()")
        contributed, prev_run = self._frames.pop()
        if contributed:
            self._eff.pop()
            self._prefix.pop()
        self._run = prev_run

    def context_id(self) -> int:
        """64-bit id of the current bounded-depth calling context (``O(1)``)."""
        m = len(self._eff)
        if m == 0:
            return 0
        n = self.depth if self.depth < m else m
        self._extend_pow(n)
        w = (self._prefix[m] - (self._prefix[m - n] * self._pow[n])) & MASK64
        return _mix64(w ^ (n * 0x9E3779B97F4A7C15))

    def exact_context(self) -> Tuple[int, ...]:
        """Exact tuple of the top-N effective call sites (collision-free shadow)."""
        m = len(self._eff)
        if m == 0:
            return ()
        n = self.depth if self.depth < m else m
        return tuple(self._eff[m - n : m])

    def context_id_recomputed(self) -> int:
        """Reference ``O(N)`` recomputation of :meth:`context_id`.

        Present so the ``O(1)`` rolling-window formula can be differentially tested
        against a straightforward implementation.
        """
        window = self.exact_context()
        n = len(window)
        if n == 0:
            return 0
        w = 0
        for c in window:
            w = ((w * self.P) + c) & MASK64
        return _mix64(w ^ (n * 0x9E3779B97F4A7C15))

    def bb(self, bb_id: Hashable) -> None:
        """Instrumentation probe: basic block entered, keyed by calling context."""
        cur = self.loc(bb_id)
        edge = self._prev_loc ^ cur
        ctx = self.context_id()
        self.bitmap.hit(edge ^ (ctx & self.bitmap.mask))
        if ctx not in self.context_trace:
            self.context_trace.add(ctx)
        if self.track_exact:
            self.exact_trace.add((self.exact_context(), self._prev_bb, bb_id))
        self._prev_loc = cur >> 1
        self._prev_bb = bb_id
        self.n_probes += 1

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        """Begin a new execution."""
        self.bitmap.reset()
        self.exact_trace = set()
        self.context_trace = set()
        self._eff.clear()
        self._prefix = [0]
        self._frames.clear()
        self._run = 0
        self._prev_loc = 0
        self._prev_bb = None

    def trace_signature(self) -> frozenset:
        """Exact per-execution element set."""
        return frozenset(self.exact_trace)

    def evaluate(self, commit: bool = True) -> Dict[str, Any]:
        """Check the current execution for context-sensitive novelty."""
        status = self.bitmap.has_new_bits(commit=commit)
        fresh_ctx = self.context_trace - self.contexts
        new_exact = 0
        if self.track_exact:
            fresh = self.exact_trace - self.exact_elements
            new_exact = len(fresh)
            if commit and fresh:
                self.exact_elements |= fresh
                self.exact_contexts |= {e[0] for e in fresh}
        if commit and fresh_ctx:
            self.contexts |= fresh_ctx
        return {
            "afl_status": status,
            "new_elements": new_exact,
            "new_contexts": len(fresh_ctx),
            "is_new": status > 0,
        }

    def stats(self) -> Dict[str, Any]:
        """Coverage, context-space and collision statistics."""
        exact = len(self.exact_elements)
        mapped = self.bitmap.covered
        n_ctx = len(self.contexts)
        n_ctx_exact = len(self.exact_contexts)
        return {
            "dimension": self.dimension,
            "context_depth": self.depth,
            "recursion_cap": self.recursion_cap,
            "map_bits": self.bitmap.map_bits,
            "distinct_contexts_hashed": n_ctx,
            "distinct_contexts_exact": n_ctx_exact,
            "context_hash_collisions": max(0, n_ctx_exact - n_ctx),
            "exact_elements": exact,
            "mapped_slots_covered": mapped,
            "collisions": max(0, exact - mapped),
            "collision_rate": (exact - mapped) / exact if exact else 0.0,
            "expected_collisions_birthday": (exact * exact) / (2 ** (self.bitmap.map_bits + 1))
            if exact
            else 0.0,
            "max_effective_stack_depth": self.max_effective_depth,
            "probe_calls": self.n_probes,
        }


# --------------------------------------------------------------------------- #
# D2 - Data-flow value-range sensitivity
# --------------------------------------------------------------------------- #
class ValueRangeTracker:
    """Dynamic value tracking at critical execution nodes.

    Two kinds of node are instrumented:

    ``value`` nodes
        Loop trip counts, buffer lengths, pointer offsets. Discretized with
        :func:`signed_log2_bucket` into 129 order-of-magnitude buckets.

    ``cmp`` nodes
        Integer comparisons. Discretized three ways so that *progress* toward
        satisfying the comparison is visible even though the branch outcome (and
        therefore the edge) has not changed:

        * :func:`distance_bucket` - ``log2 |a - b|``, a monotone gradient,
        * most-significant equal-byte prefix length (big-endian magic fields),
        * least-significant equal-byte prefix length (little-endian magic fields).

        Best-so-far distance and prefix values are retained per node, so an input
        that gets *strictly closer* to satisfying a guard is judged interesting even
        when no new bucket is entered. This is the mechanism by which multi-byte
        magic-value guards are solved byte-by-byte instead of by 2^-32 brute force.

    Sensitivity metric
    ------------------
    Per node, the Shannon entropy of the bucket histogram (plug-in and Miller-Madow
    corrected) quantifies how much value diversity the corpus has induced at that
    node; a node with near-zero entropy is one the fuzzer is failing to exercise.

    Parameters
    ----------
    map_bits : int, optional
        Log2 bitmap size for the bounded projection, default 16.
    track_exact : bool, optional
        Maintain exact element sets, default True.
    """

    dimension = "D2"

    def __init__(self, map_bits: int = 16, track_exact: bool = True) -> None:
        self.bitmap = _AFLBitmap(map_bits)
        self.track_exact = bool(track_exact)
        self.node_hist: Dict[Any, Counter] = {}
        self.exact_elements: Set[Tuple[Any, str, int]] = set()
        self.exact_trace: Set[Tuple[Any, str, int]] = set()
        self.best_distance: Dict[Any, int] = {}
        self.best_prefix: Dict[Any, int] = {}
        self._trace_distance: Dict[Any, int] = {}
        self._trace_prefix: Dict[Any, int] = {}
        self.n_probes = 0

    # -- instrumentation ---------------------------------------------------- #
    def value(self, node_id: Hashable, v: int) -> None:
        """Instrumentation probe: value ``v`` observed at value node ``node_id``."""
        b = signed_log2_bucket(v)
        el = (node_id, "mag", b)
        self.exact_trace.add(el)
        self.bitmap.hit(_mix64(hash(node_id) ^ (b * 0x9E3779B1)))
        hist = self.node_hist.get(node_id)
        if hist is None:
            hist = self.node_hist[node_id] = Counter()
        hist[b] += 1
        self.n_probes += 1

    def cmp_(self, node_id: Hashable, a: int, b: int, width: int = 4) -> None:
        """Instrumentation probe: comparison of ``a`` and ``b`` at ``node_id``.

        Parameters
        ----------
        node_id : hashable
            Static identifier of the comparison site.
        a, b : int
            Operand values.
        width : int, optional
            Operand width in bytes, default 4.
        """
        d = abs(int(a) - int(b))
        db = distance_bucket(a, b)
        pm = common_byte_prefix(a, b, width, msb_first=True)
        pl = common_byte_prefix(a, b, width, msb_first=False)
        hp = max(pm, pl)

        t = self.exact_trace
        t.add((node_id, "dist", db))
        t.add((node_id, "pmsb", pm))
        t.add((node_id, "plsb", pl))

        h = hash(node_id)
        self.bitmap.hit(_mix64(h ^ (db * 0x27220A95)))
        self.bitmap.hit(_mix64(h ^ ((pm + 1) * 0x85EBCA6B)))
        self.bitmap.hit(_mix64(h ^ ((pl + 1) * 0xC2B2AE35)))

        # laf-intel style comparison splitting: each individually matching byte
        # position is its own coverage element, so a multi-byte magic value can be
        # solved one byte at a time instead of all at once. A prefix-only signal
        # cannot reward matching, say, byte 2 of 4 while bytes 0-1 still differ.
        if width > 1:
            ua = int(a) & ((1 << (8 * width)) - 1)
            ub = int(b) & ((1 << (8 * width)) - 1)
            for i in range(width):
                if ((ua >> (8 * i)) & 0xFF) == ((ub >> (8 * i)) & 0xFF):
                    t.add((node_id, "bytehit", i))
                    self.bitmap.hit(_mix64(h ^ ((i + 1) * 0x9E3779B1) ^ 0x5BF03635))

        hist = self.node_hist.get(node_id)
        if hist is None:
            hist = self.node_hist[node_id] = Counter()
        hist[db] += 1

        cur = self._trace_distance.get(node_id)
        if cur is None or d < cur:
            self._trace_distance[node_id] = d
        curp = self._trace_prefix.get(node_id)
        if curp is None or hp > curp:
            self._trace_prefix[node_id] = hp
        self.n_probes += 1

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        """Begin a new execution."""
        self.bitmap.reset()
        self.exact_trace = set()
        self._trace_distance = {}
        self._trace_prefix = {}

    def trace_signature(self) -> frozenset:
        """Exact per-execution element set."""
        return frozenset(self.exact_trace)

    def evaluate(self, commit: bool = True) -> Dict[str, Any]:
        """Check the current execution for value-range novelty.

        Novelty is the disjunction of (i) a previously unseen ``(node, kind, bucket)``
        element and (ii) a strict improvement in best-so-far comparison distance or
        equal-byte prefix at any node.
        """
        status = self.bitmap.has_new_bits(commit=commit)
        fresh = self.exact_trace - self.exact_elements
        n_improved = 0
        for node, d in self._trace_distance.items():
            best = self.best_distance.get(node)
            if best is None or d < best:
                n_improved += 1
                if commit:
                    self.best_distance[node] = d
        for node, p in self._trace_prefix.items():
            best = self.best_prefix.get(node)
            if best is None or p > best:
                n_improved += 1
                if commit:
                    self.best_prefix[node] = p
        if commit and fresh:
            self.exact_elements |= fresh
        return {
            "afl_status": status,
            "new_elements": len(fresh),
            "n_improved_gradients": n_improved,
            "is_new": bool(fresh) or n_improved > 0,
        }

    # -- metrics ------------------------------------------------------------ #
    def node_entropy(self) -> Dict[Any, Dict[str, float]]:
        """Per-node bucket-histogram entropy (plug-in and Miller-Madow, bits)."""
        out: Dict[Any, Dict[str, float]] = {}
        for node, hist in self.node_hist.items():
            counts = list(hist.values())
            k = len(counts)
            h = shannon_entropy_bits(counts)
            out[str(node)] = {
                "n_observations": int(sum(counts)),
                "n_buckets": k,
                "entropy_bits_plugin": h,
                "entropy_bits_miller_madow": miller_madow_entropy_bits(counts),
                "normalized_entropy": (h / math.log2(k)) if k > 1 else 0.0,
            }
        return out

    def stats(self) -> Dict[str, Any]:
        """Coverage, entropy summary and collision statistics."""
        exact = len(self.exact_elements)
        mapped = self.bitmap.covered
        ent = self.node_entropy()
        hs = [v["entropy_bits_plugin"] for v in ent.values()]
        return {
            "dimension": self.dimension,
            "map_bits": self.bitmap.map_bits,
            "n_value_nodes": len(self.node_hist),
            "exact_elements": exact,
            "mapped_slots_covered": mapped,
            "collisions": max(0, exact - mapped),
            "collision_rate": (exact - mapped) / exact if exact else 0.0,
            "mean_node_entropy_bits": float(np.mean(hs)) if hs else 0.0,
            "total_node_entropy_bits": float(np.sum(hs)) if hs else 0.0,
            "n_gradient_nodes": len(self.best_distance),
            "n_solved_comparisons": int(sum(1 for d in self.best_distance.values() if d == 0)),
            "probe_calls": self.n_probes,
        }


# --------------------------------------------------------------------------- #
# D3 - Implicit state-machine transition tracking
# --------------------------------------------------------------------------- #
class StateMachineTracker:
    """Implicit API state-machine transition tracking with surprisal novelty.

    The tracker observes a sequence of API invocations and maintains a directed
    multigraph over *abstract states*. An abstract state is either supplied by the
    harness (``state_tag`` - the observable library state, e.g. handle lifecycle
    phase, as in StateAFL) or derived from the API identity and return-code class
    (as in AFLNet, which uses response codes as states). Transitions are triples
    ``(src_state, api_id, dst_state)``.

    Novelty weight
    --------------
    With ``N`` total transition observations over ``K`` distinct observed
    transitions and a Jeffreys prior ``alpha = 0.5``, an add-alpha categorical model
    with one reserved escape symbol gives

        p_hat(t) = (c_t + alpha) / (N + alpha (K + 1))

    and the novelty weight is the surprisal ``w(t) = -log2 p_hat(t)``. This is
    maximal (but finite) for a never-observed transition, decreases monotonically in
    ``c_t``, and needs no hand-tuned constants. Weights are usable directly as
    corpus-energy multipliers in later steps.

    In addition to transitions, ``k``-gram coverage of the raw API sequence is
    tracked, because some ordering-dependent bugs are invisible in the state
    projection but visible in the call sequence.

    Parameters
    ----------
    kgram : int, optional
        Length of API-sequence n-grams to track, default 3.
    alpha : float, optional
        Add-alpha smoothing parameter, default 0.5 (Jeffreys).
    map_bits : int, optional
        Log2 bitmap size, default 16.
    """

    dimension = "D3"
    INITIAL_STATE = 0

    def __init__(self, kgram: int = 3, alpha: float = 0.5, map_bits: int = 16) -> None:
        if kgram < 1:
            raise ValueError("kgram must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.kgram = int(kgram)
        self.alpha = float(alpha)
        self.bitmap = _AFLBitmap(map_bits)

        self.transition_counts: Counter = Counter()
        self.states: Set[Any] = set()
        self.kgrams: Set[Tuple[Any, ...]] = set()
        self.adjacency: Dict[Any, Set[Tuple[Any, Any]]] = {}

        self.exact_trace: Set[Tuple[Any, Any, Any]] = set()
        self._kgram_trace: Set[Tuple[Any, ...]] = set()
        self._state_trace: Set[Any] = set()
        self._seq: List[Any] = []
        self._cur_state: Any = self.INITIAL_STATE
        self._trace_weight_sum = 0.0
        self.n_probes = 0

    # -- instrumentation ---------------------------------------------------- #
    def api(
        self,
        api_id: Hashable,
        ret_class: int = 0,
        state_tag: Optional[Hashable] = None,
    ) -> None:
        """Instrumentation probe: an API function was invoked.

        Parameters
        ----------
        api_id : hashable
            API function identifier.
        ret_class : int, optional
            Discretized return-code class (AFLNet-style state signal).
        state_tag : hashable, optional
            Explicit observable abstract state after the call. When omitted, the
            destination state is derived from ``(api_id, ret_class)``.
        """
        src = self._cur_state
        dst = state_tag if state_tag is not None else ("auto", api_id, int(ret_class))
        t = (src, api_id, dst)
        self.exact_trace.add(t)
        self._state_trace.add(dst)
        self.bitmap.hit(_mix64(hash(src) * 31 ^ hash(api_id) * 131 ^ hash(dst) * 8191))
        self._trace_weight_sum += self.novelty_weight(t)

        self._seq.append(api_id)
        k = self.kgram
        if len(self._seq) >= k:
            self._kgram_trace.add(tuple(self._seq[-k:]))
        self._cur_state = dst
        self.n_probes += 1

    # -- novelty ------------------------------------------------------------ #
    def novelty_weight(self, transition: Tuple[Any, Any, Any]) -> float:
        """Surprisal ``-log2 p_hat(t)`` in bits under the smoothed model."""
        n = sum(self.transition_counts.values())
        k = len(self.transition_counts)
        denom = n + self.alpha * (k + 1)
        p = (self.transition_counts.get(transition, 0) + self.alpha) / denom
        return -math.log2(p)

    def max_novelty_weight(self) -> float:
        """Surprisal assigned to an as-yet-unobserved transition."""
        n = sum(self.transition_counts.values())
        k = len(self.transition_counts)
        denom = n + self.alpha * (k + 1)
        return -math.log2(self.alpha / denom)

    def frontier_states(self) -> List[Any]:
        """States whose observed out-degree is below the median (exploration frontier)."""
        if not self.adjacency:
            return []
        degs = sorted(len(v) for v in self.adjacency.values())
        med = degs[len(degs) // 2]
        return [s for s, v in self.adjacency.items() if len(v) < med]

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        """Begin a new execution."""
        self.bitmap.reset()
        self.exact_trace = set()
        self._kgram_trace = set()
        self._state_trace = set()
        self._seq = []
        self._cur_state = self.INITIAL_STATE
        self._trace_weight_sum = 0.0

    def trace_signature(self) -> frozenset:
        """Exact per-execution element set (transitions plus k-grams)."""
        return frozenset(self.exact_trace) | frozenset(("kg",) + g for g in self._kgram_trace)

    def evaluate(self, commit: bool = True) -> Dict[str, Any]:
        """Check the current execution for state-transition novelty."""
        status = self.bitmap.has_new_bits(commit=commit)
        fresh_t = {t for t in self.exact_trace if t not in self.transition_counts}
        fresh_g = self._kgram_trace - self.kgrams
        fresh_s = self._state_trace - self.states
        weight = self._trace_weight_sum
        if commit:
            for t in self.exact_trace:
                self.transition_counts[t] += 1
                src, api, dst = t
                self.adjacency.setdefault(src, set()).add((api, dst))
                self.adjacency.setdefault(dst, set())
            self.kgrams |= fresh_g
            self.states |= fresh_s
        return {
            "afl_status": status,
            "new_elements": len(fresh_t) + len(fresh_g),
            "new_transitions": len(fresh_t),
            "new_kgrams": len(fresh_g),
            "new_states": len(fresh_s),
            "surprisal_bits": weight,
            "is_new": bool(fresh_t or fresh_g or fresh_s),
        }

    # -- graph analysis ----------------------------------------------------- #
    def _scc_stats(self) -> Tuple[int, int]:
        """Iterative Tarjan SCC: returns ``(n_scc, largest_scc_size)``."""
        succ = {s: {d for _, d in v} for s, v in self.adjacency.items()}
        index: Dict[Any, int] = {}
        low: Dict[Any, int] = {}
        on_stack: Set[Any] = set()
        stack: List[Any] = []
        counter = 0
        sizes: List[int] = []
        for root in list(succ.keys()):
            if root in index:
                continue
            work: List[Tuple[Any, int]] = [(root, 0)]
            index[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            while work:
                node, pi = work[-1]
                kids = list(succ.get(node, ()))
                if pi < len(kids):
                    work[-1] = (node, pi + 1)
                    w = kids[pi]
                    if w not in index:
                        index[w] = low[w] = counter
                        counter += 1
                        stack.append(w)
                        on_stack.add(w)
                        work.append((w, 0))
                    elif w in on_stack:
                        if low[w] < low[node]:
                            low[node] = low[w]
                else:
                    work.pop()
                    if work:
                        parent = work[-1][0]
                        if low[node] < low[parent]:
                            low[parent] = low[node]
                    if low[node] == index[node]:
                        size = 0
                        while True:
                            w = stack.pop()
                            on_stack.discard(w)
                            size += 1
                            if w == node:
                                break
                        sizes.append(size)
        return len(sizes), (max(sizes) if sizes else 0)

    def graph_export(self) -> Dict[str, Any]:
        """Serializable view of the inferred state graph."""
        nodes = sorted({str(s) for s in self.adjacency} | {str(s) for s in self.states})
        edges = [
            {"src": str(src), "api": str(api), "dst": str(dst), "count": int(c)}
            for (src, api, dst), c in sorted(self.transition_counts.items(), key=lambda kv: -kv[1])
        ]
        return {"nodes": nodes, "edges": edges}

    def stats(self) -> Dict[str, Any]:
        """Graph, coverage and novelty statistics."""
        n_states = len(self.adjacency)
        n_trans = len(self.transition_counts)
        out_degs = [len(v) for v in self.adjacency.values()] or [0]
        self_loops = sum(1 for (s, _, d) in self.transition_counts if s == d)
        n_scc, largest = self._scc_stats()
        counts = list(self.transition_counts.values())
        exact = n_trans + len(self.kgrams)
        mapped = self.bitmap.covered
        return {
            "dimension": self.dimension,
            "kgram": self.kgram,
            "alpha": self.alpha,
            "map_bits": self.bitmap.map_bits,
            "n_abstract_states": n_states,
            "n_transitions": n_trans,
            "n_kgrams": len(self.kgrams),
            "exact_elements": exact,
            "mapped_slots_covered": mapped,
            "collision_rate": (exact - mapped) / exact if exact else 0.0,
            "mean_out_degree": float(np.mean(out_degs)),
            "max_out_degree": int(np.max(out_degs)),
            "n_self_loops": self_loops,
            "n_strongly_connected_components": n_scc,
            "largest_scc_size": largest,
            "graph_density": (n_trans / (n_states * n_states)) if n_states else 0.0,
            "transition_entropy_bits": shannon_entropy_bits(counts),
            "transition_entropy_bits_miller_madow": miller_madow_entropy_bits(counts),
            "max_novelty_weight_bits": self.max_novelty_weight(),
            "probe_calls": self.n_probes,
        }


# --------------------------------------------------------------------------- #
# Composite engine
# --------------------------------------------------------------------------- #
@dataclass
class Feedback:
    """Per-execution guidance verdict.

    Attributes
    ----------
    is_interesting : bool
        Whether any enabled dimension reported novelty (corpus-admission decision).
    fused_score : float
        Weighted, per-dimension-normalized novelty score for energy assignment.
    per_dim : dict
        Raw ``evaluate()`` payload for each enabled dimension.
    """

    is_interesting: bool = False
    fused_score: float = 0.0
    per_dim: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def new_elements(self, dim: str) -> int:
        """Number of newly discovered exact elements in ``dim`` (0 if disabled)."""
        return int(self.per_dim.get(dim, {}).get("new_elements", 0))


class MultiDimGuidanceEngine:
    """Composite 3D guidance engine (D1+D2+D3) over the D0 baseline.

    The probe API mirrors what a compiler instrumentation pass would emit, so the
    same engine object is passed to an instrumented target and drives every enabled
    dimension. Disabled dimensions are hard-gated with boolean flags so that
    ablation configurations pay (close to) no probe cost - this is what makes the
    overhead comparison against the pure-D0 baseline meaningful.

    Parameters
    ----------
    dims : sequence of str, optional
        Subset of ``("D0", "D1", "D2", "D3")`` to enable. Default: all.
    map_bits : int, optional
        Log2 bitmap size for every dimension, default 16.
    context_depth : int, optional
        Calling-context window ``N``, default 4.
    recursion_cap : int, optional
        Recursion folding cap, default 2.
    kgram : int, optional
        API-sequence n-gram length, default 3.
    weights : dict, optional
        Per-dimension fusion weights, default equal weight 1.0.
    track_exact : bool, optional
        Maintain exact shadow element sets (needed for partition analysis and
        collision accounting; adds overhead). Default True.
    """

    def __init__(
        self,
        dims: Sequence[str] = DIMENSIONS,
        map_bits: int = 16,
        context_depth: int = 4,
        recursion_cap: int = 2,
        kgram: int = 3,
        ngram: int = 3,
        weights: Optional[Dict[str, float]] = None,
        track_exact: bool = True,
    ) -> None:
        unknown = set(dims) - set(DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown dimensions: {sorted(unknown)}")
        self.dims = tuple(d for d in DIMENSIONS if d in set(dims))
        if not self.dims:
            raise ValueError("at least one dimension must be enabled")
        self.weights = {d: 1.0 for d in DIMENSIONS}
        if weights:
            self.weights.update(weights)

        self.d0 = AFLEdgeTracker(map_bits=map_bits, track_exact=track_exact)
        self.d1 = CallingContextTracker(
            depth=context_depth,
            map_bits=map_bits,
            recursion_cap=recursion_cap,
            track_exact=track_exact,
        )
        self.d2 = ValueRangeTracker(map_bits=map_bits, track_exact=track_exact)
        self.d3 = StateMachineTracker(kgram=kgram, map_bits=map_bits)
        self.dn = NGramTracker(k=ngram, map_bits=map_bits, track_exact=track_exact)

        self._e0 = "D0" in self.dims
        self._e1 = "D1" in self.dims
        self._e2 = "D2" in self.dims
        self._e3 = "D3" in self.dims
        self._en = "DN" in self.dims

        self.n_executions = 0
        self.n_interesting = 0

    # -- probe API (called from instrumented targets) ----------------------- #
    def bb(self, bb_id: Hashable) -> None:
        """Basic-block probe."""
        if self._e0:
            self.d0.bb(bb_id)
        if self._e1:
            self.d1.bb(bb_id)
        if self._en:
            self.dn.bb(bb_id)

    def enter(self, callsite_id: int) -> None:
        """Call-site entry probe."""
        if self._e1:
            self.d1.enter(callsite_id)

    def leave(self) -> None:
        """Call-site exit probe."""
        if self._e1:
            self.d1.leave()

    def value(self, node_id: Hashable, v: int) -> None:
        """Value-node probe (lengths, counters, offsets)."""
        if self._e2:
            self.d2.value(node_id, v)

    def cmp_(self, node_id: Hashable, a: int, b: int, width: int = 4) -> None:
        """Comparison-node probe."""
        if self._e2:
            self.d2.cmp_(node_id, a, b, width)

    def api(
        self,
        api_id: Hashable,
        ret_class: int = 0,
        state_tag: Optional[Hashable] = None,
    ) -> None:
        """API-invocation probe."""
        if self._e3:
            self.d3.api(api_id, ret_class, state_tag)

    # -- lifecycle ---------------------------------------------------------- #
    def _enabled_trackers(self) -> List[Any]:
        out = []
        if self._e0:
            out.append(self.d0)
        if self._e1:
            out.append(self.d1)
        if self._e2:
            out.append(self.d2)
        if self._e3:
            out.append(self.d3)
        if self._en:
            out.append(self.dn)
        return out

    def reset(self) -> None:
        """Begin a new execution (clears all per-execution traces)."""
        for t in self._enabled_trackers():
            t.reset()

    def evaluate(self, commit: bool = True) -> Feedback:
        """Aggregate per-dimension novelty into a corpus-admission verdict.

        Parameters
        ----------
        commit : bool, optional
            Claim the coverage globally when True (the normal fuzzing path); pass
            False for non-destructive inspection.

        Returns
        -------
        Feedback
        """
        fb = Feedback()
        score = 0.0
        for t in self._enabled_trackers():
            r = t.evaluate(commit=commit)
            fb.per_dim[t.dimension] = r
            if r["is_new"]:
                fb.is_interesting = True
            w = self.weights[t.dimension]
            # log1p keeps a single pathological execution from dominating energy
            score += w * math.log1p(float(r.get("new_elements", 0)))
            if t.dimension == "D2":
                score += w * math.log1p(float(r.get("n_improved_gradients", 0)))
            if t.dimension == "D3":
                score += w * 0.1 * float(r.get("surprisal_bits", 0.0))
        fb.fused_score = score
        self.n_executions += 1
        if fb.is_interesting:
            self.n_interesting += 1
        return fb

    def trace_signature(self) -> frozenset:
        """Union of per-dimension exact element sets, tagged by dimension.

        Used for partition/refinement analysis: two executions are in the same cell
        of ``Pi`` iff their signatures are equal.
        """
        sig: Set[Any] = set()
        for t in self._enabled_trackers():
            d = t.dimension
            for e in t.trace_signature():
                sig.add((d, e))
        return frozenset(sig)

    def d0_signature_counted(self) -> frozenset:
        """AFL baseline signature including hit-count classes.

        Raises
        ------
        RuntimeError
            If D0 is not enabled or exact tracking is off.
        """
        if not self._e0 or not self.d0.track_exact:
            raise RuntimeError("D0 with track_exact=True is required")
        return self.d0.trace_signature_counted()

    def stats(self) -> Dict[str, Any]:
        """Per-dimension statistics plus engine-level totals."""
        per = {t.dimension: t.stats() for t in self._enabled_trackers()}
        total_exact = sum(v.get("exact_elements", 0) for v in per.values())
        return {
            "enabled_dimensions": list(self.dims),
            "n_executions": self.n_executions,
            "n_interesting": self.n_interesting,
            "total_exact_elements": total_exact,
            "per_dimension": per,
        }
