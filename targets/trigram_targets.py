"""Trigram-dependent benchmark targets: bigram-blind, higher-order-navigable.

Step 1 proved analytically that an AFL-style coverage map is a function of the
*bigram multiset* of the executed basic-block sequence (Theorem 2), and therefore
cannot distinguish two executions whose block-pair multisets agree. That is a
statement about a blind spot. These targets are built to sit squarely inside it.

Each target reads its input as a fixed-length walk over a symbol alphabet, with
one handler basic block and one call site per symbol. Three properties follow by
construction and all three are load-bearing:

1. the executed block sequence *is* the symbol sequence, so a block ``n``-gram is
   a symbol ``n``-gram and nothing is diluted by loop headers or dispatch blocks;
2. the D1 calling-context window of depth ``N`` is exactly the trailing
   ``N``-gram of the symbol sequence, so ``N`` is directly the feedback order;
3. the walk has **fixed length**, so every execution runs the same number of
   blocks and no hit-count class can leak progress. (An earlier variable-length
   design handed the baseline a free length gradient through AFL's saturating
   count classes; that is a real trap and it is why the length is fixed here.)

Progress toward the bug is advanced by *table lookup only* - never by a branch -
so no partial progress creates a new edge, a new hit-count class, or any other
control-flow event. The single branch in each target is the final bug predicate.

The three targets isolate three different escape routes from the bigram blind
spot:

* :class:`TG1TrigramLock` - pure order-3. The bug needs a *set* of five specific
  symbol trigrams. Crucially, the constituent *bigrams* of those trigrams are
  cheap to collect, so an edge-coverage fuzzer climbs quickly to a saturation
  plateau and then has zero remaining gradient, while order-``>=3`` feedback
  keeps climbing one byte at a time. This is the positive half of the bigram
  characterisation, made operational.
* :class:`TG2ValueTrigram` - order-3 gate *unlocking* a value-range gate. Neither
  D1 nor D2 alone suffices: the 32-bit magic comparison is not even reached until
  the trigram set is complete, and the trigram set is worthless without solving
  the magic. Measures whether a scheduler can hold a two-phase objective.
* :class:`TG3SurprisalMaze` - a chain of rare state transitions. Representing
  the chain as op history needs order 6, a context space of ``8^6``; D3
  represents the identical distinction in seven states. The target measures
  compactness of abstraction rather than raw expressiveness, which is where the
  dilution cost actually bites. Its call stack is deliberately shallow, so D1
  contributes nothing and the target isolates D3.

Every target has a byte-identical native C implementation in
``native/trigram_target_lib.c``; :func:`check_trigram_fidelity` cross-validates
the decision traces so that the Python models used in the experiments are
certified stand-ins for compiled code rather than assumed ones.
"""

from __future__ import annotations

import ctypes
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "TRACE_CAP",
    "TargetResult",
    "TG1TrigramLock",
    "TG2ValueTrigram",
    "TG3SurprisalMaze",
    "TRIGRAM_TARGETS",
    "get_trigram_targets",
    "SEED_INPUT_BY_TARGET",
    "build_trigram_lib",
    "NativeTrigramLib",
    "check_trigram_fidelity",
    "bigram_multiset",
    "trigram_multiset",
    "decode_walk",
]

# --- artifact-evaluation path bootstrap (replaces the session-root anchor) --- #
import sys as _ae_sys  # noqa: E402
from pathlib import Path as _AEPath  # noqa: E402

AE_ROOT = _AEPath(__file__).resolve().parent.parent
if str(AE_ROOT) not in _ae_sys.path:
    _ae_sys.path.insert(0, str(AE_ROOT))
import ae_paths as _ae_paths  # noqa: E402,F401  (registers src/ and targets/)

SESSION = AE_ROOT
# --- end bootstrap --------------------------------------------------------- #
NATIVE_SRC = AE_ROOT / "native" / "trigram_target_lib.c"
NATIVE_LIB = AE_ROOT / "native" / "libtrigram_target.so"

#: Maximum decision-trace length exchanged with the native library.
TRACE_CAP = 8192


@dataclass
class TargetResult:
    """Outcome of one target execution.

    Attributes
    ----------
    trace : list of int
        Decision trace: the sequence of basic-block ids executed. This is the
        object compared against the native implementation.
    milestone : int
        Monotone partial-progress level. Pure harness instrumentation - it emits
        no trace point and is therefore invisible to every coverage dimension
        equally, so it can be used to measure progress without granting any
        configuration a signal the others lack.
    bug : bool
        Whether the seeded bug oracle fired.
    """

    trace: List[int] = field(default_factory=list)
    milestone: int = 0
    bug: bool = False


class _Probeless:
    """No-op probe sink, so a target can run without a guidance engine."""

    def bb(self, _bb_id) -> None:  # noqa: D102
        return

    def enter(self, _cs) -> None:  # noqa: D102
        return

    def leave(self) -> None:  # noqa: D102
        return

    def value(self, _node, _v) -> None:  # noqa: D102
        return

    def cmp_(self, _node, _a, _b, width: int = 4) -> None:  # noqa: D102
        return

    def api(self, _api_id, ret_class: int = 0, state_tag=None) -> None:  # noqa: D102
        return


_NOPROBE = _Probeless()


# --------------------------------------------------------------------------- #
# Sequence helpers (used by the targets and by the blind-spot analysis)
# --------------------------------------------------------------------------- #
def decode_walk(data: bytes, walk: int, xor: int, mask: int) -> Tuple[int, ...]:
    """Decode ``data`` into the fixed-length symbol walk a target executes.

    Parameters
    ----------
    data : bytes
        Input buffer; indices wrap modulo ``len(data)``.
    walk : int
        Number of walk steps.
    xor : int
        Decode XOR applied before masking.
    mask : int
        Alphabet mask (``ALPHABET - 1``).

    Returns
    -------
    tuple of int
        Symbol sequence of length ``walk``; empty when ``data`` is empty.
    """
    if not data:
        return ()
    n = len(data)
    return tuple((data[i % n] ^ xor) & mask for i in range(walk))


def bigram_multiset(seq: Sequence[int]) -> Tuple[Tuple[Tuple[int, int], int], ...]:
    """Canonical, hashable bigram multiset of ``seq``."""
    counts: Dict[Tuple[int, int], int] = {}
    for a, b in zip(seq, seq[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return tuple(sorted(counts.items()))


def trigram_multiset(
    seq: Sequence[int],
) -> Tuple[Tuple[Tuple[int, int, int], int], ...]:
    """Canonical, hashable trigram multiset of ``seq``."""
    counts: Dict[Tuple[int, int, int], int] = {}
    for a, b, c in zip(seq, seq[1:], seq[2:]):
        counts[(a, b, c)] = counts.get((a, b, c), 0) + 1
    return tuple(sorted(counts.items()))


def _popcount(x: int) -> int:
    """Population count of a small non-negative integer."""
    return bin(x).count("1")


def _overlapping_trigrams(word: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    """The overlapping trigrams of ``word``, in order.

    Defined at module scope rather than inline in a class body because a
    comprehension inside a class body cannot see the class's own attributes -
    a scoping rule that silently turns ``REQ = tuple(... KEY_WORD ...)`` into a
    ``NameError`` at import time.

    Parameters
    ----------
    word : sequence of int
        Key word of length at least 3.

    Returns
    -------
    tuple of (int, int, int)
        ``len(word) - 2`` trigrams, in ladder order.

    Raises
    ------
    ValueError
        If ``word`` is shorter than three symbols.
    """
    if len(word) < 3:
        raise ValueError("a trigram ladder needs a key word of at least 3 symbols")
    return tuple(tuple(word[i : i + 3]) for i in range(len(word) - 2))


def _build_advtab(
    req: Sequence[Tuple[int, int, int]], alphabet: int
) -> Tuple[Tuple[int, ...], ...]:
    """Branchless ordered-ladder advance table: ``advtab[prog][trigram]``.

    Progress is ``prog += advtab[prog][tri]``, which is 1 exactly when ``tri`` is
    the trigram the ladder is currently waiting for and 0 otherwise. Row
    ``len(req)`` is all zeros, so the top of the ladder is absorbing and the
    counter cannot run off the end.

    An earlier design required the trigrams as an unordered *set*. A pilot showed
    why that was the wrong shape: acquiring the trigrams at scattered walk
    positions raised the milestone without moving the input any closer to the
    bug, because closing the last rung then demanded global consistency across
    the whole walk rather than one more byte. Progress became a needle rather
    than a ladder, and every condition - baseline and treatment alike - stalled
    one rung short. Requiring the trigrams *in order at increasing positions*
    restores the property the benchmark needs: each rung is reachable from the
    one below by a local mutation, so the difference between conditions is
    whether the feedback function *retains the intermediate seed*, which is
    exactly the question under study.

    The table is indexed with a fixed 4-bit-per-symbol packing,
    ``(a << 8) | (b << 4) | c``, independently of the alphabet size. Keeping the
    packing width fixed while the alphabet is retuned costs a few unused
    kilobytes and removes a class of silent C/Python index drift that the
    fidelity check would otherwise have to catch after the fact.

    Parameters
    ----------
    req : sequence of (int, int, int)
        Required trigrams, in the order the ladder demands them.
    alphabet : int
        Symbol alphabet size; at most 16 so every symbol fits the packing.

    Returns
    -------
    tuple of tuple of int
        ``len(req) + 1`` rows of ``16 ** 3`` entries each.

    Raises
    ------
    ValueError
        If ``alphabet`` exceeds 16 or a required symbol is out of range.
    """
    if not 2 <= alphabet <= 16:
        raise ValueError("the 4-bit trigram packing requires 2 <= alphabet <= 16")
    if any(not 0 <= x < alphabet for tri in req for x in tri):
        raise ValueError("required trigram symbols must lie inside the alphabet")
    rows: List[Tuple[int, ...]] = []
    for j in range(len(req) + 1):
        row = [0] * (16**3)
        if j < len(req):
            a, b, c = req[j]
            row[(a << 8) | (b << 4) | c] = 1
        rows.append(tuple(row))
    return tuple(rows)


# --------------------------------------------------------------------------- #
# TG1 - pure order-3 trigram lock
# --------------------------------------------------------------------------- #
class TG1TrigramLock:
    """Bug requires four symbol trigrams *in order*; their bigrams are decoys.

    Construction
    ------------
    The walk has :data:`WALK` steps over a :data:`ALPHABET`-symbol alphabet.
    Three symbols matter: ``A``, ``B``, ``C``. The key word is ``A B A C A B``,
    whose four trigrams ``ABA``, ``BAC``, ``ACA``, ``CAB`` are exactly
    :data:`REQ_TRIGRAMS`. A monotone counter advances one rung each time the
    trigram the ladder is *currently waiting for* appears, at any later walk
    position. The bug fires at the top of the ladder. Advancing is
    ``prog += ADVTAB[prog][trigram]``, a table lookup with no branch, so no rung
    is observable in control flow.

    Why the order matters
    ---------------------
    An earlier version required the four trigrams as an unordered *set*. That
    version discriminated nothing, and the reason is worth recording: acquiring
    trigrams at scattered walk positions raised the milestone without moving the
    input closer to the bug, so closing the last rung demanded global consistency
    across the whole walk instead of one more byte. Progress was a needle, not a
    ladder, and every condition - baseline and treatment alike - stalled one rung
    short. Under the ordered ladder each rung is reachable from the one below by
    a local mutation, so what separates the conditions is whether the feedback
    function *retains the intermediate seed*. That is the question Step 2 exists
    to answer, and the set version could not ask it.

    Why an edge-coverage fuzzer stalls
    ----------------------------------
    The key word's *bigram* multiset is ``{AB:2, BA:1, AC:1, CA:1}``. Those four
    block pairs are individually cheap - each is one lucky two-byte coincidence -
    so an AFL-style fuzzer collects all of them early and its edge map saturates.
    From that point every rearrangement that climbs a *trigram* rung re-uses
    bigrams the map already holds, and produces no new edge, no new hit-count
    class, and hence no reward. The witness is explicit: :data:`WITNESS_NEG`
    (``A C A B A B``) is a different Eulerian trail of the *same* transition
    multigraph as :data:`WITNESS_POS` (``A B A C A B``). It has an identical
    bigram multiset, identical per-symbol execution counts, and therefore - by
    the Step 1 characterisation - a bit-for-bit identical AFL map including count
    classes. It nevertheless stalls at rung 1 while its twin fires the bug. The
    analysis script verifies each of those claims against the running engine
    rather than asserting them.

    Why order-``>=3`` feedback proceeds
    -----------------------------------
    Each rung is a specific three-symbol window at a later position, most of
    whose symbols the previous rung already pinned - typically a single byte of
    the input. The objective is a four-rung ladder with about one byte per rung,
    but the rungs are visible only to an abstraction of order 3 or more. This is
    the cleanest separation the bigram characterisation permits: same search
    space, same mutation operators, and the entire difference is whether the
    feedback function can see the rung.
    """

    NAME = "TG1_trigram_lock"
    #: Symbol alphabet; one handler basic block and one call site per symbol.
    #: Eight, not sixteen. The alphabet fixes the D1 context space at
    #: ``ALPHABET ** N``; at 16 symbols and depth 4 that is 65,536, and a pilot
    #: admitted 19,000 of 50,000 executions to the corpus, so every condition
    #: drowned in undifferentiated novelty and the schedulers were compared on
    #: noise. At 8 symbols the depth-4 space is 4,096, which the budget
    #: saturates - and saturation is the regime in which a scheduling difference
    #: is observable at all. Difficulty is supplied by the LENGTH of the ladder
    #: instead of by the width of the alphabet; see :data:`REQ_TRIGRAMS`.
    ALPHABET = 8
    MASK = ALPHABET - 1
    #: Byte -> symbol decode XOR. Not decoration: the havoc mutator's
    #: interesting-value table contains 0x00, 0x01, 0x02, 0x7F, 0x80, 0xFF, which
    #: decode through this XOR and mask to symbols {0, 2, 3, 5}. The load-bearing
    #: symbols are drawn from the complement {1, 4, 6, 7}, so the benchmark
    #: cannot be solved by the mutator's constant table instead of by feedback.
    DECODE_XOR = 0x5A
    #: fixed number of walk steps (14 trigram positions for an 8-rung ladder)
    WALK = 16
    SYM_A = 6
    SYM_B = 1
    SYM_C = 4
    #: the key word ``A B A C A B C B A C``, whose eight trigrams form the ladder
    WITNESS_POS: Tuple[int, ...] = (
        SYM_A, SYM_B, SYM_A, SYM_C, SYM_A, SYM_B, SYM_C, SYM_B, SYM_A, SYM_C,
    )
    #: Bigram-identical, non-triggering counterpart ``A C A C B A B A B C``,
    #: found by exhaustive search over the three-symbol subalphabet for a
    #: sequence with the same bigram multiset AND the same per-symbol counts as
    #: :data:`WITNESS_POS`. Same multiset of transitions means a bit-for-bit
    #: identical AFL map, count classes included; it nevertheless stalls at rung
    #: 1 while its twin fires the bug.
    WITNESS_NEG: Tuple[int, ...] = (
        SYM_A, SYM_C, SYM_A, SYM_C, SYM_B, SYM_A, SYM_B, SYM_A, SYM_B, SYM_C,
    )
    #: The eight required trigrams, in the order the ladder demands them: the
    #: overlapping trigrams of the key word. A long shallow ladder rather than a
    #: short steep one is the load-bearing choice. Each rung costs roughly one
    #: byte, so it is individually easy; eight of them in sequence is reachable
    #: only if the fuzzer *retains and re-fuzzes* every intermediate seed, which
    #: is exactly the corpus-retention and energy-allocation question Step 2
    #: exists to measure. A short ladder over a wide alphabet makes each rung
    #: expensive instead, which measures luck.
    #: Six rungs, from the key word's first eight symbols. Calibrated by pilot:
    #: an eight-rung ladder left every condition censored at rung 4 of 8, which
    #: is a floor effect and measures nothing; six rungs put the conditions in
    #: the band where they separate. The two dropped rungs remain in
    #: :data:`WITNESS_POS` as slack, so the witness pair is unaffected.
    REQ_TRIGRAMS: Tuple[Tuple[int, int, int], ...] = _overlapping_trigrams(
        WITNESS_POS
    )[:6]
    MAX_MILESTONE = len(REQ_TRIGRAMS)
    TARGETED_DIMENSION = "D1 order >= 3"

    #: branchless ordered-ladder advance table, built once at class definition
    ADVTAB: Tuple[Tuple[int, ...], ...] = _build_advtab(REQ_TRIGRAMS, ALPHABET)

    def run(self, data: bytes, probe=None) -> TargetResult:
        """Execute the target on ``data``, emitting guidance probes."""
        p = probe if probe is not None else _NOPROBE
        res = TargetResult()
        self._t(res, p, 0x01)
        if not data:
            self._t(res, p, 0x42)
            return res
        state = [0, 0, 0]  # [prog, prev_sym, prev2_sym]
        p.enter(0xE000)
        self._step(data, 0, state, res, p)
        p.leave()
        if state[0] >= len(self.REQ_TRIGRAMS):
            self._t(res, p, 0x70)
            res.bug = True
        self._t(res, p, 0x02)
        return res

    @staticmethod
    def _t(res: TargetResult, p, tid: int) -> None:
        """Trace point == basic-block probe."""
        res.trace.append(tid)
        p.bb(tid)

    def _step(self, data: bytes, i: int, state: List[int], res: TargetResult, p) -> None:
        """Consume one symbol and recurse; the recursion *is* the walk.

        Exactly one basic block executes per step - the symbol's own handler - so
        the executed block sequence is the symbol sequence verbatim and the D1
        context window of depth ``N`` is the trailing symbol ``N``-gram.
        """
        if i >= self.WALK:
            self._t(res, p, 0x41)
            return
        s = (data[i % len(data)] ^ self.DECODE_XOR) & self.MASK
        p.enter(0xE010 + s)
        self._t(res, p, 0x50 + s)
        # ---- branchless advance: one table lookup, never a branch -----------
        if i >= 2:
            tri = (state[2] << 8) | (state[1] << 4) | s
            state[0] += self.ADVTAB[state[0]][tri]
        state[2] = state[1]
        state[1] = s
        if state[0] > res.milestone:
            res.milestone = state[0]
        self._step(data, i + 1, state, res, p)
        p.leave()


# --------------------------------------------------------------------------- #
# TG2 - order-3 gate unlocking a value-range gate
# --------------------------------------------------------------------------- #
class TG2ValueTrigram:
    """A trigram lock that *unlocks* a 32-bit magic comparison.

    Two phases in strict sequence. Phase one is an ordered two-rung trigram
    ladder built exactly like :class:`TG1TrigramLock` from the key word
    ``A B A C`` (trigrams ``ABA`` then ``BAC``). Phase two reads a little-endian 32-bit word from a fixed offset
    and compares it against :data:`MAGIC`; the bug needs all four bytes.

    The comparison probe is emitted *only once phase one is complete*, so the
    two phases are genuinely sequential rather than two independent objectives
    solved in parallel. That is the point of the target: D1 alone gets through
    the trigram lock and then stalls on a 2^32 magic; D2 alone has a byte-wise
    gradient on the magic that it never gets to use, because the comparison site
    is unreachable. Only a scheduler that keeps *both* kinds of seed alive - and
    reallocates energy when the useful dimension changes mid-run - solves both.

    Phase two is instrumented with :meth:`cmp_`, whose ``bytehit`` elements are
    laf-intel style comparison splitting: each individually matching byte position
    is its own coverage element, so the magic is solvable one byte at a time
    instead of all at once. The byte-match count feeds the milestone but emits
    **no trace point**, so it grants no configuration an edge-coverage signal.
    """

    NAME = "TG2_value_trigram"
    #: eight symbols, for the same context-space reason as TG1
    ALPHABET = 8
    MASK = ALPHABET - 1
    DECODE_XOR = 0x5A
    #: 8 trigram positions for a 4-rung phase-one ladder
    WALK = 10
    SYM_A = 6
    SYM_B = 1
    SYM_C = 4
    #: offset of the little-endian 32-bit magic operand
    VAL_OFF = 8
    #: The magic constant. No byte of it appears in the mutator's interesting
    #: table, so the 32-bit gate must be solved through the D2 byte-split
    #: gradient rather than stumbled into by a constant substitution.
    MAGIC = 0x5AC31E7B
    #: key word ``A B A C A B``, whose four trigrams form the phase-one ladder
    WITNESS_POS: Tuple[int, ...] = (SYM_A, SYM_B, SYM_A, SYM_C, SYM_A, SYM_B)
    #: bigram-identical, non-triggering counterpart
    WITNESS_NEG: Tuple[int, ...] = (SYM_A, SYM_C, SYM_A, SYM_B, SYM_A, SYM_B)
    #: the four required trigrams, in ladder order
    #: three rungs, then the magic; calibrated by pilot alongside TG1
    REQ_TRIGRAMS: Tuple[Tuple[int, int, int], ...] = _overlapping_trigrams(
        WITNESS_POS
    )[:3]
    #: 3 trigram rungs + 4 magic-byte rungs
    MAX_MILESTONE = 7
    TARGETED_DIMENSION = "D1 order >= 3 then D2 value range"

    ADVTAB: Tuple[Tuple[int, ...], ...] = _build_advtab(REQ_TRIGRAMS, ALPHABET)

    def run(self, data: bytes, probe=None) -> TargetResult:
        """Execute the target on ``data``, emitting guidance probes."""
        p = probe if probe is not None else _NOPROBE
        res = TargetResult()
        self._t(res, p, 0x01)
        if not data:
            self._t(res, p, 0x42)
            return res
        state = [0, 0, 0]
        p.enter(0xE100)
        self._step(data, 0, state, res, p)
        p.leave()

        if state[0] < len(self.REQ_TRIGRAMS):
            self._t(res, p, 0x43)  # phase one incomplete: comparison unreachable
            self._t(res, p, 0x02)
            return res

        self._t(res, p, 0x44)  # phase two entered
        if len(data) < self.VAL_OFF + 4:
            self._t(res, p, 0x45)  # operand truncated
            self._t(res, p, 0x02)
            return res

        v = int.from_bytes(data[self.VAL_OFF : self.VAL_OFF + 4], "little")
        # Guidance-only probes: not part of the decision trace, so they are
        # invisible to the fidelity comparison and to every non-D2 condition.
        p.cmp_(0xE1C0, v, self.MAGIC, width=4)
        p.value(0xE1C1, v)

        nbytes = 0
        for k in range(4):
            nbytes += int(((v >> (8 * k)) & 0xFF) == ((self.MAGIC >> (8 * k)) & 0xFF))
        ms = len(self.REQ_TRIGRAMS) + nbytes
        if ms > res.milestone:
            res.milestone = ms

        if v == self.MAGIC:
            self._t(res, p, 0x70)
            res.bug = True
        self._t(res, p, 0x02)
        return res

    @staticmethod
    def _t(res: TargetResult, p, tid: int) -> None:
        """Trace point == basic-block probe."""
        res.trace.append(tid)
        p.bb(tid)

    def _step(self, data: bytes, i: int, state: List[int], res: TargetResult, p) -> None:
        """Consume one symbol and recurse (see :meth:`TG1TrigramLock._step`)."""
        if i >= self.WALK:
            self._t(res, p, 0x41)
            return
        s = (data[i % len(data)] ^ self.DECODE_XOR) & self.MASK
        p.enter(0xE110 + s)
        self._t(res, p, 0x50 + s)
        if i >= 2:
            tri = (state[2] << 8) | (state[1] << 4) | s
            state[0] += self.ADVTAB[state[0]][tri]
        state[2] = state[1]
        state[1] = s
        if state[0] > res.milestone:
            res.milestone = state[0]
        self._step(data, i + 1, state, res, p)
        p.leave()


# --------------------------------------------------------------------------- #
# TG3 - rare-transition chain, compact under D3
# --------------------------------------------------------------------------- #
class TG3SurprisalMaze:
    """A six-link chain of rare state transitions over an 8-op alphabet.

    Each of :data:`NOPS` steps decodes one op from the input and advances a state
    machine by table lookup. The machine has :data:`NSTATES` states; from state
    ``s`` the single op :data:`ADVANCE` ``[s]`` moves to ``s + 1`` and every other
    op returns to state 0. Reaching the final state fires the bug.

    Why the call stack is deliberately shallow
    ------------------------------------------
    Unlike TG1 and TG2, the op handlers here are *iterative*: each ``enter`` is
    matched by a ``leave`` inside the same loop iteration, so the D1 context
    stack never exceeds depth two and the D1 window depth ``N`` has no effect on
    this target whatsoever. That is a design choice, not an oversight, and it is
    what makes TG3 a clean D3 isolator: any advantage a 3D condition shows here
    is attributable to D2 or D3, because D1 is provably constant across depths.
    A pilot confirms the prediction exactly - the ``N = 4`` and ``N = 16``
    conditions produce bit-identical trials on this target.

    What the target is actually about
    ---------------------------------
    The chain is a length-6 property of the op sequence. An abstraction that
    tracks op history would need order 6 to represent it, at a context space of
    ``8^6 = 262,144``. D3 represents the identical distinction in
    :data:`NSTATES` states, because the chain position *is* the sufficient
    statistic of the history. The contrast is about the **size** of the
    abstraction that suffices, not about which abstraction is more expressive -
    and size is what the corpus pays for, which is precisely why the Step 1
    depth sweep saw ``N = 16`` lose to ``N = 4`` despite strictly refining it.

    D3's surprisal signal is what makes the chain climbable. A transition that
    advances the chain is, by construction, rare under the Jeffreys-smoothed
    transition model the tracker maintains, so ``S = -log2 P(tau)`` is large for
    exactly the transitions worth pursuing. The state is exposed to the tracker
    through the ``state_tag`` argument of the ``api`` probe, which is what a real
    protocol fuzzer obtains from a response code.

    The op handlers do emit one basic block each, so edge coverage sees the op
    sequence and is not artificially blinded; D0 simply cannot represent where in
    the chain an execution got to, because the advance is branchless.
    """

    NAME = "TG3_surprisal_maze"
    #: op alphabet size
    NOPS_ALPHA = 8
    OP_MASK = NOPS_ALPHA - 1
    DECODE_XOR = 0x3C
    #: number of decoded ops per execution
    NOPS = 12
    #: number of abstract states, including the absorbing final state
    NSTATES = 7
    #: ADVANCE[s] is the unique op that moves state s to s + 1
    ADVANCE: Tuple[int, ...] = (6, 3, 5, 1, 6, 2, 0)
    #: the op chain that walks state 0 to the absorbing state; used by the
    #: fidelity sampler so the bug branch is actually exercised on both sides.
    WITNESS_OPS: Tuple[int, ...] = ADVANCE[: NSTATES - 1]
    MAX_MILESTONE = 6
    TARGETED_DIMENSION = "D3 surprisal state machine"

    def run(self, data: bytes, probe=None) -> TargetResult:
        """Execute the target on ``data``, emitting guidance probes."""
        p = probe if probe is not None else _NOPROBE
        res = TargetResult()
        self._t(res, p, 0x01)
        if not data:
            self._t(res, p, 0x42)
            return res
        n = len(data)
        state = 0
        p.enter(0xE200)
        for i in range(self.NOPS):
            op = (data[i % n] ^ self.DECODE_XOR) & self.OP_MASK
            p.enter(0xE210 + op)
            self._t(res, p, 0x50 + op)
            # ---- branchless advance: arithmetic, never a branch -------------
            adv = int(op == self.ADVANCE[state])
            # state -> state+1 on the advancing op, else back to 0; the final
            # state is absorbing so the bug latches once reached.
            fin = int(state >= self.NSTATES - 1)
            state = fin * state + (1 - fin) * (adv * (state + 1))
            # Guidance-only D3 probe: the abstract state is observable, which is
            # what a protocol fuzzer gets from a response code.
            p.api(0xE220 + op, ret_class=state, state_tag=("tg3", state))
            if state > res.milestone:
                res.milestone = state
            p.leave()
        p.leave()
        if state >= self.NSTATES - 1:
            self._t(res, p, 0x70)
            res.bug = True
        self._t(res, p, 0x02)
        return res

    @staticmethod
    def _t(res: TargetResult, p, tid: int) -> None:
        """Trace point == basic-block probe."""
        res.trace.append(tid)
        p.bb(tid)


TRIGRAM_TARGETS: Tuple[type, ...] = (TG1TrigramLock, TG2ValueTrigram, TG3SurprisalMaze)


def get_trigram_targets() -> List[Any]:
    """Instantiate one of each trigram target, in canonical order."""
    return [cls() for cls in TRIGRAM_TARGETS]


#: Per-target seed input. Length is chosen so the whole walk (and, for TG2, the
#: magic operand) is addressable without the modulo wrap aliasing walk positions
#: onto one another, which would silently reduce the search dimension.
SEED_INPUT_BY_TARGET: Dict[str, bytes] = {
    TG1TrigramLock.NAME: bytes(16),
    TG2ValueTrigram.NAME: bytes(16),
    TG3SurprisalMaze.NAME: bytes(16),
}


# --------------------------------------------------------------------------- #
# Native bridge
# --------------------------------------------------------------------------- #
def build_trigram_lib(force: bool = False) -> Path:
    """Compile ``native/trigram_target_lib.c`` into ``data/libtrigram_target.so``.

    Parameters
    ----------
    force : bool, optional
        Rebuild even when the shared object is newer than the source.

    Returns
    -------
    pathlib.Path
        Path to the compiled shared library.

    Raises
    ------
    RuntimeError
        If compilation fails; the compiler diagnostics are included verbatim.
    """
    NATIVE_LIB.parent.mkdir(parents=True, exist_ok=True)
    if (
        not force
        and NATIVE_LIB.exists()
        and NATIVE_LIB.stat().st_mtime >= NATIVE_SRC.stat().st_mtime
    ):
        return NATIVE_LIB
    cmd = [
        "gcc",
        "-O2",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-o",
        str(NATIVE_LIB),
        str(NATIVE_SRC),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gcc failed:\n{proc.stdout}\n{proc.stderr}")
    return NATIVE_LIB


class NativeTrigramLib:
    """ctypes bridge to the compiled native trigram benchmark library.

    Parameters
    ----------
    lib_path : pathlib.Path, optional
        Path to the shared object; built on demand when omitted.
    """

    _ENTRIES = {
        TG1TrigramLock.NAME: "tg1_entry",
        TG2ValueTrigram.NAME: "tg2_entry",
        TG3SurprisalMaze.NAME: "tg3_entry",
    }

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        self.path = Path(lib_path) if lib_path else build_trigram_lib()
        self.lib = ctypes.CDLL(str(self.path))
        self._fn: Dict[str, Any] = {}
        for name, sym in self._ENTRIES.items():
            fn = getattr(self.lib, sym)
            fn.restype = ctypes.c_int
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            self._fn[name] = fn
        self._tr = (ctypes.c_uint * TRACE_CAP)()
        self._trlen = ctypes.c_int(0)
        self._ms = ctypes.c_int(0)
        self.compiler_version = subprocess.run(
            ["gcc", "--version"], capture_output=True, text=True, check=False
        ).stdout.splitlines()[0]

    def run(self, target_name: str, data: bytes) -> TargetResult:
        """Execute a native target and return its decision trace.

        Parameters
        ----------
        target_name : str
            One of the keys of :attr:`_ENTRIES`.
        data : bytes
            Input buffer.

        Returns
        -------
        TargetResult
        """
        fn = self._fn[target_name]
        n = len(data)
        payload = data if n else b"\x00"
        buf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        bug = fn(
            buf,
            ctypes.c_size_t(n),
            self._tr,
            TRACE_CAP,
            ctypes.byref(self._trlen),
            ctypes.byref(self._ms),
        )
        ln = min(self._trlen.value, TRACE_CAP)
        return TargetResult(
            trace=list(self._tr[:ln]), milestone=int(self._ms.value), bug=bool(bug)
        )


def check_trigram_fidelity(
    n_inputs: int = 5000, seed: int = 20260811, max_len: int = 96
) -> Dict[str, Any]:
    """Cross-validate the Python trigram models against the native C library.

    Random inputs of varied length are pushed through both implementations and
    the decision trace, milestone and bug flag are compared *exactly*. Any
    mismatch invalidates the use of the Python models as stand-ins for native
    code, so the report records the first few divergences verbatim rather than
    only a count.

    The input distribution is deliberately mixed. Uniform random bytes almost
    never satisfy a trigram lock, so a uniform-only sample would certify the
    implementations agree on the *failing* path and say nothing about the
    triggering one. A share of the inputs is therefore drawn from the key word
    and from single-byte perturbations of it, which exercises the milestone
    ladder and the bug branch on both sides.

    Parameters
    ----------
    n_inputs : int, optional
        Inputs per target, default 5000.
    seed : int, optional
        RNG seed, default 20260811.
    max_len : int, optional
        Maximum input length, default 96.

    Returns
    -------
    dict
        Per-target comparison counts, coverage of the milestone ladder, and any
        mismatches.
    """
    lib = NativeTrigramLib()
    rng = random.Random(seed)
    targets = get_trigram_targets()
    report: Dict[str, Any] = {
        "n_inputs_per_target": int(n_inputs),
        "seed": int(seed),
        "library": str(lib.path),
        "compiler": lib.compiler_version,
        "targets": {},
        "total_mismatches": 0,
    }

    for tgt in targets:
        mismatches: List[Dict[str, Any]] = []
        ms_hist: Dict[int, int] = {}
        n_bug = 0
        for _ in range(n_inputs):
            data = _fidelity_input(rng, tgt, max_len)
            py = tgt.run(data, None)
            c = lib.run(tgt.NAME, data)
            ms_hist[py.milestone] = ms_hist.get(py.milestone, 0) + 1
            n_bug += int(py.bug)
            if py.trace != c.trace or py.milestone != c.milestone or py.bug != c.bug:
                if len(mismatches) < 5:
                    mismatches.append(
                        {
                            "input_hex": data.hex(),
                            "py_trace": py.trace[:64],
                            "c_trace": c.trace[:64],
                            "py_milestone": py.milestone,
                            "c_milestone": c.milestone,
                            "py_bug": py.bug,
                            "c_bug": c.bug,
                        }
                    )
        report["targets"][tgt.NAME] = {
            "n_inputs": int(n_inputs),
            "n_mismatches": len(mismatches),
            "mismatches": mismatches,
            "milestone_histogram": {str(k): v for k, v in sorted(ms_hist.items())},
            "max_milestone_seen": max(ms_hist) if ms_hist else 0,
            "n_bug_triggering_inputs": n_bug,
        }
        report["total_mismatches"] += len(mismatches)

    report["all_match"] = report["total_mismatches"] == 0
    return report


def _fidelity_input(rng: random.Random, tgt: Any, max_len: int) -> bytes:
    """Draw one fidelity-test input, mixing uniform and near-solution samples.

    A uniform-only sample would leave the triggering path of every target
    untested, so 40% of draws are built from the target's own key word and then
    perturbed, which walks the milestone ladder including its top rung.
    """
    # The key sequence and the mask that encodes it differ per target family.
    key = getattr(tgt, "WITNESS_POS", None) or getattr(tgt, "WITNESS_OPS", None)
    mask = getattr(tgt, "MASK", None)
    if mask is None:
        mask = getattr(tgt, "OP_MASK", 0xFF)

    mode = rng.random()
    if mode < 0.60 or key is None:
        ln = rng.randint(0, max_len)
        return bytes(rng.getrandbits(8) for _ in range(ln))

    # Near-solution: encode the key sequence into the walk positions, then
    # perturb. The encoding inverts the decode exactly - for any XOR ``X``,
    # ``(((b & ~M) | s) ^ X) ^ X & M == s`` - so it is target-agnostic.
    ln = 16
    buf = bytearray(rng.getrandbits(8) for _ in range(ln))
    off = rng.randrange(0, max(1, ln - len(key))) if rng.random() < 0.3 else 0
    for i, s in enumerate(key):
        if off + i >= ln:
            break
        buf[off + i] = ((buf[off + i] & ~mask & 0xFF) | s) ^ tgt.DECODE_XOR
    if getattr(tgt, "NAME", "") == TG2ValueTrigram.NAME and rng.random() < 0.5:
        magic = TG2ValueTrigram.MAGIC.to_bytes(4, "little")
        keep = rng.randint(0, 4)
        for k in range(keep):
            buf[TG2ValueTrigram.VAL_OFF + k] = magic[k]
    # single-byte perturbation, so the sample straddles the bug predicate
    if rng.random() < 0.5 and buf:
        j = rng.randrange(len(buf))
        buf[j] = rng.getrandbits(8)
    return bytes(buf)
