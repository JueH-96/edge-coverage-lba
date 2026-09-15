"""
Benchmark targets for Step 1 validation of the 3D Multidimensional Guidance Engine.

Each target exists in two forms:

1. **Native C** (``native/target_lib.c``, compiled to ``data/libtarget.so``) - the
   ground-truth implementation, reached through :class:`NativeTargetLib` via ctypes.
2. **Instrumented Python model** (this module) - the same control flow, carrying the
   guidance probes (``bb``/``enter``/``leave``/``cmp_``/``value``/``api``) that a
   compiler instrumentation pass would insert.

Both forms emit an identical *decision trace* (the ordered sequence of trace-point
ids executed) plus a milestone level and a bug flag. :func:`check_fidelity`
cross-validates them over thousands of random inputs; this is what licenses using the
instrumented Python models as stand-ins for the native library when measuring
guidance quality.

Design principle
----------------
A seeded bug only discriminates between coverage metrics if the *partial progress*
toward it is invisible to the weaker metric. Any ``if`` on a progress condition
immediately leaks that progress to edge coverage, because taking the branch is itself
a new edge. So every intermediate state update below is **unconditional data flow**,
and the only branch depending on the full precondition is the bug check. Edge
coverage saturates within a handful of random inputs and then supplies no gradient at
all; the extra dimensions must supply it.

Targets
-------
``T1_context`` (targets D1)
    Recursive TLV parser with delta-encoded container headers over a shared 48-byte
    scratch buffer, so the layout depends on the *order* of container types along the
    path from the root. ``SEQ MAP SEQ MAP`` overflows while ``SEQ SEQ MAP MAP`` is
    safe, yet the two execute the same basic blocks the *same number of times* -
    indistinguishable to AFL edge coverage including its hit-count classes. (An
    earlier depth-based design was discarded precisely because AFL's hit-count
    bucketing does leak recursion depth and the baseline solved it.) Bounded-depth
    calling contexts make each container path a distinct element.

``T2_value_range`` (targets D2)
    Three chained 16-bit equality gates (0xC0DE, 0x1234, 0xBEEF) plus an XOR
    checksum. Blind search needs ~2^48 executions; edge coverage gives one
    taken/not-taken pair per gate and no gradient. Comparison-distance and
    equal-byte-prefix bucketing turn each gate into a byte-at-a-time hill climb.

``T3_state_machine`` (targets D3)
    Handle lifecycle over a 16-operation API alphabet. ``configure`` caches an interior pointer
    unconditionally, ``flush`` commits the current mode unconditionally, ``close``
    forgets to invalidate the cache, ``read``/``seek`` re-validate it, and ``write``
    consumes the commit. The bug needs
    ``open -> configure(mode=5) -> flush -> close -> open -> write`` with no
    invalidating operation in between. No ordering prefix produces a new edge.

Milestones are harness instrumentation, not target logic: they update a progress
counter, emit no trace point, and are therefore invisible to every coverage
dimension equally.

Seeded bugs use explicit oracles (a bounds predicate, an allocation-generation
counter) instead of actually corrupting memory, so runs stay deterministic and safe;
ASan/MSan would supply the same oracles on a production target.
"""

from __future__ import annotations

import ctypes
import random
import subprocess
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

SESSION_DIR = AE_ROOT
# --- end bootstrap --------------------------------------------------------- #
NATIVE_SRC = AE_ROOT / "native" / "target_lib.c"
NATIVE_LIB = AE_ROOT / "native" / "libtarget.so"
TRACE_CAP = 8192

__all__ = [
    "TargetResult",
    "T1ContextTarget",
    "T2ValueRangeTarget",
    "T3StateMachineTarget",
    "T4TrigramTarget",
    "ALL_TARGETS",
    "get_targets",
    "NativeTargetLib",
    "build_native_lib",
    "check_fidelity",
]


# --------------------------------------------------------------------------- #
# Result container / null probe
# --------------------------------------------------------------------------- #
@dataclass
class TargetResult:
    """Outcome of one target execution.

    Attributes
    ----------
    trace : list of int
        Ordered decision-trace point ids (ground truth for fidelity checking).
    milestone : int
        Monotone progress level toward the seeded bug (0 = no progress).
    bug : bool
        Whether the seeded-bug oracle fired.
    """

    trace: List[int] = field(default_factory=list)
    milestone: int = 0
    bug: bool = False


class _Probeless:
    """No-op probe object, so a target can run without a guidance engine attached."""

    __slots__ = ()

    def bb(self, _bb_id) -> None:  # noqa: D102
        pass

    def enter(self, _cs) -> None:  # noqa: D102
        pass

    def leave(self) -> None:  # noqa: D102
        pass

    def value(self, _node, _v) -> None:  # noqa: D102
        pass

    def cmp_(self, _node, _a, _b, width: int = 4) -> None:  # noqa: D102
        pass

    def api(self, _api_id, ret_class: int = 0, state_tag=None) -> None:  # noqa: D102
        pass


_NOPROBE = _Probeless()


# --------------------------------------------------------------------------- #
# T1 - calling-context (recursion depth) dependent scratch overflow
# --------------------------------------------------------------------------- #
class T1ContextTarget:
    """Recursive TLV parser whose scratch layout depends on the container *path*.

    Grammar (``tag = byte & 0x3F``): ``60`` INT (4 bytes), ``61`` STR (length byte +
    payload), ``62`` SEQ, ``63`` MAP (count byte + recursive elements), anything else
    END. The 64-value tag space (only four values meaningful, as in ASN.1 or
    protobuf-style formats) is what makes a 4-container alternating path rare under
    blind search while costing a guided fuzzer only ~64 tries per level.

    Headers are delta-encoded: a container nested in a container of the **same**
    type shares its parent's header and costs ``ADV_SAME`` bytes of the shared
    48-byte scratch buffer, while a type **switch** needs a fresh header costing
    ``ADV_SWITCH`` bytes. The overflow therefore depends on the *order* of container
    types, not on depth or on how many of each type appear::

        SEQ MAP SEQ MAP SEQ MAP  ->  6 * 8      = 48   -> overflows for any len >= 1
        SEQ SEQ SEQ SEQ SEQ SEQ  ->  8 + 5 * 2  = 18   -> always safe

    Those two inputs execute the same basic blocks the *same number of times*, so
    they are indistinguishable to AFL edge coverage including its saturating
    hit-count classes - which do leak recursion depth and would otherwise hand the
    baseline the progress signal.

    ``off + len > 48`` with ``len <= 8`` requires ``off >= 41``, i.e. at least
    **five type switches**. AFL edge coverage implicitly carries *one* level of
    calling context (the edge from a container body block into ``parse_value``
    identifies the parent type), so requiring a single alternation is not enough: an
    earlier 4-switch version of this target was solved by the edge-coverage baseline
    in 3/3 pilot trials. Five switches puts the required path beyond one level of
    implicit context, beyond hit-count bucketing, and beyond a shallow (N=4) context
    window - which is what the context-depth sweep measures. ``MAXDEPTH`` is 8, so
    same-type nesting alone can never reach the overflow (``8 + 2*7 = 22``).

    Milestones: ``min(off // 8, 6)`` as the path accumulates header cost;
    milestone 7 is the bug.
    """

    NAME = "T1_context"
    MAX_MILESTONE = 7
    SCRATCH = 48
    ADV_SWITCH = 8
    ADV_SAME = 2
    LEN_LIMIT = 8
    MAXDEPTH = 8
    MAXCOUNT = 4
    MS_CAP = 6
    TAG_MASK = 0x3F
    TAG_INT = 60
    TAG_STR = 61
    TAG_SEQ = 62
    TAG_MAP = 63
    TYPE_NONE = 0
    #: dimension expected to supply the missing gradient (for reporting only)
    TARGETED_DIMENSION = "D1"

    def run(self, data: bytes, probe=None) -> TargetResult:
        """Execute the target on ``data``, emitting guidance probes.

        Parameters
        ----------
        data : bytes
            Fuzzer input.
        probe : object, optional
            Guidance engine exposing the instrumentation probe API.

        Returns
        -------
        TargetResult
        """
        p = probe if probe is not None else _NOPROBE
        res = TargetResult()
        scratch = bytearray(self.SCRATCH + 256)
        state = {"pos": 0}

        self._t(res, p, 0x01)
        if len(data) >= 1:
            p.enter(0xA000)  # call site: entry -> parse_value
            self._parse_value(data, state, self.TYPE_NONE, 0, 0, res, p, scratch)
            p.leave()
        else:
            self._t(res, p, 0x02)
        return res

    @staticmethod
    def _t(res: TargetResult, p, tid: int) -> None:
        """Trace point == basic-block probe."""
        res.trace.append(tid)
        p.bb(tid)

    # -- parser ------------------------------------------------------------- #
    def _parse_value(self, data, state, parent_type, off, depth, res, p, scratch) -> int:
        n = len(data)
        self._t(res, p, 0x40)
        p.value(("t1", "depth"), depth)
        if depth > self.MAXDEPTH:
            self._t(res, p, 0x41)
            return -1
        if state["pos"] >= n:
            self._t(res, p, 0x42)
            return -1
        tag = data[state["pos"]] & self.TAG_MASK
        state["pos"] += 1
        if tag == self.TAG_INT:
            self._t(res, p, 0x44)
            p.enter(0xA001)  # call site: parse_value -> parse_int
            rc = self._parse_int(data, state, res, p)
            p.leave()
            return rc
        if tag == self.TAG_STR:
            self._t(res, p, 0x45)
            p.enter(0xA002)  # call site: parse_value -> parse_str
            rc = self._parse_str(data, state, off, res, p, scratch)
            p.leave()
            return rc
        if tag == self.TAG_SEQ:
            self._t(res, p, 0x46)
            p.enter(0xA005)  # call site: parse_value -> parse_seq
            rc = self._parse_container(
                data, state, self.TAG_SEQ, parent_type, off, depth, res, p, scratch,
                0x30, 0xA007,
            )
            p.leave()
            return rc
        if tag == self.TAG_MAP:
            self._t(res, p, 0x47)
            p.enter(0xA006)  # call site: parse_value -> parse_map
            rc = self._parse_container(
                data, state, self.TAG_MAP, parent_type, off, depth, res, p, scratch,
                0x50, 0xA008,
            )
            p.leave()
            return rc
        self._t(res, p, 0x43)
        return 0

    def _parse_int(self, data, state, res, p) -> int:
        n = len(data)
        self._t(res, p, 0x10)
        if state["pos"] + 4 > n:
            self._t(res, p, 0x11)
            return -1
        state["pos"] += 4
        self._t(res, p, 0x12)
        return 0

    def _parse_str(self, data, state, off, res, p, scratch) -> int:
        n = len(data)
        self._t(res, p, 0x20)
        if state["pos"] >= n:
            self._t(res, p, 0x21)
            return -1
        length = data[state["pos"]]
        state["pos"] += 1
        self._t(res, p, 0x22)
        p.value(("t1", "str_len"), length)
        p.cmp_(("t1", "len_limit"), length, self.LEN_LIMIT, width=1)
        if length > self.LEN_LIMIT:
            self._t(res, p, 0x23)
            return -1
        if length == 0:
            self._t(res, p, 0x24)
            return 0

        if state["pos"] < n and data[state["pos"]] == 0x5A:
            self._t(res, p, 0x25)

        p.cmp_(("t1", "scratch_bound"), off + length, self.SCRATCH, width=2)
        if off + length > self.SCRATCH:
            self._t(res, p, 0x2F)
            res.bug = True
            res.milestone = max(res.milestone, self.MS_CAP + 1)
            return -2

        w = off
        for _ in range(length):
            if state["pos"] >= n:
                break
            scratch[w] = data[state["pos"]] ^ 0x11
            state["pos"] += 1
            w += 1
        self._t(res, p, 0x26)
        return 0

    def _parse_container(
        self, data, state, self_type, parent_type, off, depth, res, p, scratch,
        base_id, child_callsite,
    ) -> int:
        """Shared SEQ/MAP body; ``base_id`` keeps their trace points distinct."""
        n = len(data)
        self._t(res, p, base_id + 0)
        if state["pos"] >= n:
            self._t(res, p, base_id + 1)
            return -1
        cnt = data[state["pos"]]
        state["pos"] += 1
        if cnt > self.MAXCOUNT:
            cnt = self.MAXCOUNT

        # unconditional data flow: the header cost never appears as a branch
        new_off = off + (self.ADV_SAME if self_type == parent_type else self.ADV_SWITCH)
        # harness-only progress proxy; emits no trace point
        res.milestone = max(res.milestone, min(new_off // self.ADV_SWITCH, self.MS_CAP))
        self._t(res, p, base_id + 2)
        p.value(("t1", "scratch_off"), new_off)
        p.value(("t1", "list_count"), cnt)

        for _ in range(cnt):
            if state["pos"] >= n:
                self._t(res, p, base_id + 3)
                break
            p.enter(child_callsite)  # call site: container -> parse_value
            rc = self._parse_value(
                data, state, self_type, new_off, depth + 1, res, p, scratch
            )
            p.leave()
            if rc < 0:
                self._t(res, p, base_id + 4)
                return -1
            if res.bug:
                return -2
        self._t(res, p, base_id + 5)
        return 0


# --------------------------------------------------------------------------- #
# T2 - chained magic-value gates
# --------------------------------------------------------------------------- #
class T2ValueRangeTarget:
    """Three chained 16-bit magic-value gates plus an XOR-checksum gate.

    Milestones equal the number of gates satisfied (0..4); milestone 4 is the bug.
    Blind search needs ~2^48 executions, so a fuzzer without a value-distance signal
    is expected to stall at milestone 0.
    """

    NAME = "T2_value_range"
    MAX_MILESTONE = 4
    MAGIC = 0xC0DE
    VER = 0x1234
    TAG = 0xBEEF
    TARGETED_DIMENSION = "D2"

    def run(self, data: bytes, probe=None) -> TargetResult:
        """Execute the target on ``data``, emitting guidance probes."""
        p = probe if probe is not None else _NOPROBE
        res = TargetResult()
        n = len(data)

        def t(tid: int) -> None:
            res.trace.append(tid)
            p.bb(tid)

        p.enter(0xB000)
        p.api("t2_parse", 0, state_tag=("t2", 0))
        t(0x01)
        p.value(("t2", "n"), n)
        if n < 16:
            t(0x02)
            p.leave()
            return res

        magic = data[0] | (data[1] << 8)
        p.cmp_(("t2", "magic"), magic, self.MAGIC, width=2)
        if magic != self.MAGIC:
            t(0x10)
            p.leave()
            return res
        res.milestone = 1
        t(0x11)
        p.api("t2_gate1", 0, state_tag=("t2", 1))

        ver = data[2] | (data[3] << 8)
        p.cmp_(("t2", "ver"), ver, self.VER, width=2)
        if ver != self.VER:
            t(0x12)
            p.leave()
            return res
        res.milestone = 2
        t(0x13)
        p.api("t2_gate2", 0, state_tag=("t2", 2))

        tag = data[4] | (data[5] << 8)
        p.cmp_(("t2", "tag"), tag, self.TAG, width=2)
        if tag != self.TAG:
            t(0x14)
            p.leave()
            return res
        res.milestone = 3
        t(0x15)
        p.api("t2_gate3", 0, state_tag=("t2", 3))

        declared = data[6]
        length = data[7]
        if length > n - 8:
            length = n - 8
        p.value(("t2", "chk_len"), length)
        chk = 0
        for i in range(length):
            chk ^= data[8 + i]
        p.cmp_(("t2", "checksum"), chk, declared, width=1)
        if chk != declared:
            t(0x16)
            p.leave()
            return res

        res.milestone = 4
        res.bug = True
        t(0x1F)
        p.api("t2_bug", 1, state_tag=("t2", 4))
        p.leave()
        return res


# --------------------------------------------------------------------------- #
# T3 - ordering-dependent stale pointer
# --------------------------------------------------------------------------- #
class T3StateMachineTarget:
    """Handle lifecycle whose seeded bug depends only on API ordering.

    Input is consumed two bytes at a time: ``op = byte0 & 0x0F`` selects one of 16
    operations (``open``, ``configure``, ``write``, ``close``, ``read``, ``flush``,
    ``seek``, ``stat``, and eight ``ioctl`` variants) and ``byte1`` is the argument.
    Every handle-touching filler operation re-resolves the cached interior pointer,
    so gaps in the required ordering are destructive - which is what makes the
    ordering hard for blind search (~1e-7 per random input) yet climbable one state
    transition at a time.

    All intermediate state updates are unconditional assignments, so no ordering
    prefix creates a new edge. The abstract state handed to D3 is a *mechanical*
    projection of the handle's observable scalar fields (StateAFL-style state
    snapshot) - it encodes no knowledge of where the bug is.

    Milestones: 1 = open succeeded, 2 = configure armed the cache (mode 5),
    3 = flush committed the armed mode, 4 = reopen with a stale cache, 5 = bug.
    """

    NAME = "T3_state_machine"
    MAX_MILESTONE = 5
    MAXOPS = 16
    BUFCAP = 16
    ARM_MODE = 5
    TARGETED_DIMENSION = "D3"

    def run(self, data: bytes, probe=None) -> TargetResult:
        """Execute the target on ``data``, emitting guidance probes."""
        p = probe if probe is not None else _NOPROBE
        res = TargetResult()
        n = len(data)

        def t(tid: int) -> None:
            res.trace.append(tid)
            p.bb(tid)

        h_open = 0
        gen = 0
        mode = 0
        commit_mode = 0
        cached = 0
        cached_gen = -1
        buf_alive = False

        def tag() -> Tuple[Any, ...]:
            """Mechanical snapshot of the handle's observable scalar state.

            Includes a pointer-freshness bit obtained by comparing the cached
            pointer's allocation generation against the live one - exactly what a
            memory-snapshot differ observes when the cached pointer no longer
            matches the current buffer address.
            """
            return (
                "t3",
                h_open,
                mode,
                commit_mode,
                cached,
                1 if (cached and cached_gen != gen) else 0,
            )

        t(0x01)
        nops = 0
        i = 0
        while i + 1 < n and nops < self.MAXOPS:
            op = data[i] & 0x0F
            arg = data[i + 1]
            i += 2
            nops += 1
            p.value(("t3", "op_index"), nops)

            if op == 0:  # open
                p.enter(0xC000)
                t(0x10)
                if h_open:
                    t(0x11)
                    p.api("open", 1, state_tag=tag())
                    p.leave()
                    continue
                gen += 1
                buf_alive = True
                h_open = 1
                t(0x12)
                res.milestone = max(res.milestone, 1)
                if cached and cached_gen != gen:
                    res.milestone = max(res.milestone, 4)
                p.api("open", 0, state_tag=tag())
                p.leave()

            elif op == 1:  # configure
                p.enter(0xC001)
                t(0x20)
                if not h_open:
                    t(0x21)
                    p.api("configure", 1, state_tag=tag())
                    p.leave()
                    continue
                mode = arg & 0x07
                cached = 1
                cached_gen = gen
                t(0x22)
                p.value(("t3", "mode"), mode)
                if mode == self.ARM_MODE:
                    res.milestone = max(res.milestone, 2)
                p.api("configure", 0, state_tag=tag())
                p.leave()

            elif op == 5:  # flush
                p.enter(0xC005)
                t(0x50)
                if not h_open:
                    t(0x51)
                    p.api("flush", 1, state_tag=tag())
                    p.leave()
                    continue
                commit_mode = mode
                t(0x52)
                p.value(("t3", "commit_mode"), commit_mode)
                if commit_mode == self.ARM_MODE:
                    res.milestone = max(res.milestone, 3)
                p.api("flush", 0, state_tag=tag())
                p.leave()

            elif op == 2:  # write
                p.enter(0xC002)
                t(0x30)
                if not h_open:
                    t(0x31)
                    p.api("write", 1, state_tag=tag())
                    p.leave()
                    continue
                use_cached = (commit_mode == self.ARM_MODE) and cached
                p.cmp_(("t3", "gen"), cached_gen, gen, width=4)
                if use_cached and cached_gen != gen:
                    t(0x3F)
                    res.bug = True
                    res.milestone = max(res.milestone, 5)
                    p.api("write", 2, state_tag=("t3", "USE_AFTER_FREE"))
                    p.leave()
                    break
                commit_mode = 0
                t(0x32)
                p.api("write", 0, state_tag=tag())
                p.leave()

            elif op == 3:  # close
                p.enter(0xC003)
                t(0x40)
                if not h_open:
                    t(0x41)
                    p.api("close", 1, state_tag=tag())
                    p.leave()
                    continue
                buf_alive = False
                h_open = 0
                # BUG SOURCE: `cached` / `cached_gen` deliberately NOT invalidated.
                t(0x42)
                p.api("close", 0, state_tag=tag())
                p.leave()

            elif op == 4:  # read
                p.enter(0xC004)
                t(0x60)
                if not h_open:
                    t(0x61)
                    p.api("read", 1, state_tag=tag())
                    p.leave()
                    continue
                cached_gen = gen
                t(0x62)
                p.api("read", 0, state_tag=tag())
                p.leave()

            elif op == 6:  # seek
                p.enter(0xC006)
                t(0x70)
                if not h_open:
                    t(0x71)
                    p.api("seek", 1, state_tag=tag())
                    p.leave()
                    continue
                cached_gen = gen
                commit_mode = 0
                t(0x72)
                p.api("seek", 0, state_tag=tag())
                p.leave()

            elif op == 7:  # stat
                p.enter(0xC007)
                t(0x80)
                p.api("stat", 0, state_tag=tag())
                p.leave()

            else:  # ioctl 8..15
                p.enter(0xC008)
                t(0x90)
                if not h_open:
                    t(0x91)
                    p.api("ioctl", 1, state_tag=tag())
                    p.leave()
                    continue
                cached_gen = gen  # re-resolves the cache
                t(0x92)
                p.api("ioctl", 0, state_tag=tag())
                p.leave()

        t(0x02)
        _ = buf_alive
        return res


# --------------------------------------------------------------------------- #
# T4 - genuine higher-order (n-gram) blind spot
# --------------------------------------------------------------------------- #
def _kmp_automaton(pattern: Sequence[int], alphabet: int) -> Tuple[Tuple[int, ...], ...]:
    """Build the Knuth-Morris-Pratt matching automaton for ``pattern``.

    The returned table ``delta[k][s]`` gives the next match state after reading
    symbol ``s`` in state ``k``; state ``len(pattern)`` is absorbing. Because the
    table is a pure array lookup, advancing the automaton in the target costs *no
    branch*, so the match state never leaks into control flow.

    Parameters
    ----------
    pattern : sequence of int
        Symbol sequence to be matched.
    alphabet : int
        Alphabet size.

    Returns
    -------
    tuple of tuple of int
        Transition table with ``len(pattern) + 1`` rows.
    """
    m = len(pattern)
    fail = [0] * (m + 1)
    k = 0
    for i in range(1, m):
        while k and pattern[i] != pattern[k]:
            k = fail[k]
        if pattern[i] == pattern[k]:
            k += 1
        fail[i + 1] = k
    rows: List[Tuple[int, ...]] = []
    for st in range(m + 1):
        row = []
        for s in range(alphabet):
            if st == m:
                row.append(m)  # absorbing
                continue
            j = st
            while j and pattern[j] != s:
                j = fail[j]
            row.append(j + 1 if pattern[j] == s else 0)
        rows.append(tuple(row))
    return tuple(rows)


class T4TrigramTarget:
    """Order-``L`` block-sequence gate: a *genuine* higher-order blind spot.

    Motivation
    ----------
    :class:`T1ContextTarget` was built to isolate calling context, but its
    triggering predicate turns out to be a function of adjacent block pairs, so it
    exercises the *no-go* half of the bigram characterisation rather than the
    positive half. T4 exists to exercise the positive half. Its triggering
    predicate is an order-6 property of the executed basic-block sequence, hence
    provably not a function of the bigram multiset, hence - by Theorem 2 -
    provably invisible to an AFL-style map however large the map and however finely
    its hit counts are classified.

    Construction
    ------------
    The input is read as a **fixed-length walk** of exactly :data:`WALK` steps.
    Step ``i`` consumes ``data[i % len(data)]`` and reduces it to a symbol
    ``(b ^ 0x5A) & (ALPHABET - 1)`` over a 32-symbol alphabet; each symbol dispatches to its
    own handler basic block and is entered through its own call site. Three
    consequences follow, and all three are load-bearing:

    1. the executed block sequence *is* the symbol sequence;
    2. the D1 calling-context window of depth ``N`` is exactly the trailing
       ``N``-gram of that sequence, and the AFL++ ``NGRAM-N`` window is the same
       ``N``-gram obtained from the block stream instead of the stack;
    3. **every execution executes exactly the same number of blocks**, so no
       hit-count class ever varies with progress. The earlier variable-length
       design was discarded for exactly this reason: AFL's saturating hit-count
       classes leak walk length, which handed the baseline a free gradient. Under
       the fixed-length walk the D0 signal is the *pair multiset alone*, which is
       what the theorem is about.

    A Knuth-Morris-Pratt automaton over the symbol alphabet advances a match state
    by **table lookup only**. The bug fires when the state reaches ``len(REQ)``,
    i.e. when the walk contains :data:`REQ` as a contiguous subsequence. Because
    the automaton is branchless, no prefix of ``REQ`` produces a new edge, a new
    hit-count class, or any other control-flow event whatsoever: partial progress
    is invisible to every abstraction of order below the prefix length.

    The witness pair
    ----------------
    ``REQ`` is ``A B C A B A`` over symbols ``A = 3``, ``B = 12``, ``C = 19``. The
    walk ``A B A B C A`` is a different Eulerian trail of the *same* multigraph of
    transitions: it has the identical bigram multiset ``{AB x2, BA, BC, CA}``, the
    identical block-execution counts, and therefore - by Theorem 2 - a bit-for-bit
    identical AFL map including hit-count classes. Its trigram multiset differs
    (``{ABC, BCA, CAB, ABA}`` against ``{ABA, BAB, ABC, BCA}``) and it does not
    trigger the bug. This is the pair the reviewer of a bigram characterisation
    should ask for, and the analysis script verifies every one of those claims
    against the running implementation rather than asserting them.

    Predicted order threshold
    -------------------------
    Order-``n`` feedback can distinguish walk prefixes only up to length ``n``, so
    it can climb the first ``n`` symbols one at a time and must then guess the
    remaining ``L - n``, at 32 tries per symbol. Against this, the cost of reaching
    a *specific* ``n``-prefix grows with the size of the ``n``-gram space the
    fuzzer must work through. The two terms move in opposite directions, which
    predicts an **interior optimum in the feedback order** - the same
    aliasing-versus-dilution trade-off the context-depth sweep sees, but here with
    the order of the triggering property known by construction.
    """

    NAME = "T4_trigram"
    #: symbol alphabet: one handler basic block per symbol
    ALPHABET = 32
    #: byte -> symbol decoding mask. The XOR is not decoration: an earlier version
    #: used ``b & 31`` with ``REQ`` over symbols 0, 1, 2, and the mutator's
    #: interesting-value table contains 0x00, 0x01 and 0x02, so havoc handed the
    #: *blind* configuration the required sequence in 1 pilot trial out of 5. The
    #: benchmark was measuring the mutator's constant table, not the coverage
    #: metric. Decoding through a XOR and placing REQ on symbols that no
    #: interesting value decodes to removes that channel.
    DECODE_XOR = 0x5A
    #: fixed number of walk steps
    WALK = 6
    #: required contiguous symbol sequence, "A B C A B" with A=3, B=12, C=19.
    #: The order L = 5 of the triggering property and the alphabet m = 32 were
    #: chosen together from the cost model of the theory section; see the design
    #: section for the calibration argument and the discarded parameterisations.
    REQ: Tuple[int, ...] = (3, 12, 19, 3, 12)
    #: bigram-equivalent, trigram-distinct, non-triggering counterpart "A B A B C"
    WITNESS_NEG: Tuple[int, ...] = (3, 12, 3, 12, 19, 3)
    MAX_MILESTONE = 5
    TARGETED_DIMENSION = "order >= 3 (D1 depth / NGRAM-k)"

    #: branchless KMP transition table, built once at class definition time
    TRANS: Tuple[Tuple[int, ...], ...] = _kmp_automaton(REQ, ALPHABET)

    def run(self, data: bytes, probe=None) -> TargetResult:
        """Execute the target on ``data``, emitting guidance probes."""
        p = probe if probe is not None else _NOPROBE
        res = TargetResult()
        self._t(res, p, 0x01)
        n = len(data)
        if n == 0:
            self._t(res, p, 0x42)
            return res
        state = [0]
        p.enter(0xD000)  # call site: entry -> step
        self._step(data, 0, state, res, p)
        p.leave()
        self._t(res, p, 0x02)
        return res

    @staticmethod
    def _t(res: TargetResult, p, tid: int) -> None:
        """Trace point == basic-block probe."""
        res.trace.append(tid)
        p.bb(tid)

    def _step(self, data, i: int, state, res: TargetResult, p) -> None:
        """Consume one symbol and recurse; the recursion *is* the walk.

        Exactly one basic block is executed per step - the symbol's own handler -
        so the executed block sequence is the symbol sequence *verbatim*. An
        earlier version emitted a loop-header block between symbols, which
        interleaved a constant block into the sequence and thereby halved the
        effective order of every n-gram abstraction: the block trigram
        ``(header, s_i, header)`` carries no more information than the symbol
        alone, and ``(s_i, header, s_{i+1})`` is a symbol *bigram* wearing a
        trigram's clothes. Instrumentation granularity changes what an order-n
        abstraction means, which is a trap worth naming explicitly.
        """
        if i >= self.WALK:
            self._t(res, p, 0x41)
            return
        s = (data[i % len(data)] ^ self.DECODE_XOR) & (self.ALPHABET - 1)
        p.enter(0xD010 + s)  # per-symbol call site: context window == n-gram
        self._t(res, p, 0x50 + s)
        # ---- branchless progress: a table lookup, never a branch -------------
        state[0] = self.TRANS[state[0]][s]
        if state[0] > res.milestone:
            res.milestone = state[0]
        if state[0] >= len(self.REQ):
            self._t(res, p, 0x70)
            res.bug = True
        self._step(data, i + 1, state, res, p)
        p.leave()


ALL_TARGETS: Tuple[type, ...] = (
    T1ContextTarget,
    T2ValueRangeTarget,
    T3StateMachineTarget,
    T4TrigramTarget,
)


def get_targets() -> List[Any]:
    """Instantiate one of each benchmark target.

    Returns
    -------
    list
        Fresh target instances in canonical order (T1, T2, T3, T4).
    """
    return [cls() for cls in ALL_TARGETS]


# --------------------------------------------------------------------------- #
# Native library bridge
# --------------------------------------------------------------------------- #
def build_native_lib(force: bool = False) -> Path:
    """Compile ``native/target_lib.c`` into a shared library.

    Parameters
    ----------
    force : bool, optional
        Rebuild even if the ``.so`` is newer than the source.

    Returns
    -------
    pathlib.Path
        Path to the compiled shared library.

    Raises
    ------
    RuntimeError
        If compilation fails.
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


class NativeTargetLib:
    """ctypes bridge to the compiled native benchmark library.

    Parameters
    ----------
    lib_path : pathlib.Path, optional
        Path to the shared library; built on demand when omitted.
    """

    _ENTRIES = {
        "T1_context": "t1_entry",
        "T2_value_range": "t2_entry",
        "T3_state_machine": "t3_entry",
        "T4_trigram": "t4_entry",
    }

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        self.path = Path(lib_path) if lib_path else build_native_lib()
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
            One of ``T1_context``, ``T2_value_range``, ``T3_state_machine``.
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


def check_fidelity(
    n_inputs: int = 5000, seed: int = 20260804, max_len: int = 96
) -> Dict[str, Any]:
    """Cross-validate the Python target models against the native C library.

    Random inputs of varied length are pushed through both implementations and the
    decision trace, milestone and bug flag are compared exactly. Any mismatch
    invalidates the use of the Python models as stand-ins for native code, so the
    report records the first few divergences verbatim.

    Parameters
    ----------
    n_inputs : int, optional
        Number of random inputs per target, default 5000.
    seed : int, optional
        RNG seed, default 20260804.
    max_len : int, optional
        Maximum random input length, default 96.

    Returns
    -------
    dict
        Per-target mismatch counts, milestone/bug rates over the random sample, and
        the compiler version used.
    """
    native = NativeTargetLib()
    targets = {t.NAME: t for t in get_targets()}
    report: Dict[str, Any] = {
        "n_inputs_per_target": n_inputs,
        "seed": seed,
        "max_input_len": max_len,
        "compiler": native.compiler_version,
        "native_lib": str(native.path),
        "per_target": {},
    }

    total_mismatches = 0
    for name, tgt in targets.items():
        # Independent RNG per target so each sees the same input distribution.
        rng = random.Random(seed)
        mismatches: List[Dict[str, Any]] = []
        n_mismatch = 0
        n_bug = 0
        ms_hist: Dict[int, int] = {}
        for _ in range(n_inputs):
            ln = rng.randint(0, max_len)
            data = bytes(rng.getrandbits(8) for _ in range(ln))
            nat = native.run(name, data)
            pyr = tgt.run(data)
            ms_hist[pyr.milestone] = ms_hist.get(pyr.milestone, 0) + 1
            if pyr.bug:
                n_bug += 1
            if (
                nat.trace != pyr.trace
                or nat.milestone != pyr.milestone
                or nat.bug != pyr.bug
            ):
                n_mismatch += 1
                if len(mismatches) < 3:
                    mismatches.append(
                        {
                            "input_hex": data.hex(),
                            "native_trace": nat.trace[:64],
                            "python_trace": pyr.trace[:64],
                            "native_milestone": nat.milestone,
                            "python_milestone": pyr.milestone,
                            "native_bug": nat.bug,
                            "python_bug": pyr.bug,
                        }
                    )
        total_mismatches += n_mismatch
        report["per_target"][name] = {
            "n_mismatches": n_mismatch,
            "mismatch_examples": mismatches,
            "random_bug_rate": n_bug / n_inputs,
            "milestone_histogram": {str(k): v for k, v in sorted(ms_hist.items())},
        }

    report["total_mismatches"] = total_mismatches
    report["fidelity_ok"] = total_mismatches == 0
    return report
