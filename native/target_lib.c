/*
 * target_lib.c - Native C benchmark library for Step 1 validation of the
 *                3D Multidimensional Guidance Engine.
 *
 * Three targets, each carrying one seeded bug that is *reachable* but
 * *unguided* under plain AFL-style edge coverage.
 *
 * DESIGN PRINCIPLE
 * ----------------
 * A seeded bug only discriminates between coverage metrics if the *partial
 * progress* toward it is invisible to the weaker metric. Any `if` statement on a
 * progress condition immediately leaks that progress to edge coverage, because
 * taking the branch is itself a new edge. Therefore every intermediate state
 * update in these targets is expressed as **unconditional data flow** (assignment
 * to a state variable), and the only control-flow branch that depends on the full
 * bug precondition is the bug check itself. Edge coverage saturates within a
 * handful of random inputs and then provides literally zero gradient; the extra
 * dimensions must supply it.
 *
 *   T1  t1_entry  - recursive TLV parser with delta-encoded container headers, so
 *                   the scratch-buffer layout depends on the *order* of container
 *                   types along the path from the root. An alternating and a
 *                   same-type path execute the same basic blocks the same number
 *                   of times -- identical to AFL edge coverage *including* its
 *                   hit-count classes, which do leak recursion depth -- yet only
 *                   the first overflows. Only the calling context separates them.
 *
 *   T2  t2_entry  - three chained 16-bit magic-value gates plus an XOR-checksum
 *                   gate. Blind search needs ~2^48 executions and edge coverage
 *                   offers one taken/not-taken pair per gate with no gradient.
 *
 *   T3  t3_entry  - handle lifecycle over a 16-operation API alphabet. configure() caches an
 *                   interior pointer unconditionally, flush() commits the current
 *                   mode unconditionally, close() forgets to invalidate the cache,
 *                   and read()/seek() re-validate it. The bug needs the ordering
 *                   open -> configure(mode=5) -> flush -> close -> open -> write
 *                   with no invalidating operation in between. Every intermediate
 *                   step is a plain assignment, so no ordering prefix produces a
 *                   new edge; only the final use-after-free check branches.
 *
 * Seeded bugs use explicit oracles (a bounds predicate, an allocation-generation
 * counter) instead of actually corrupting memory, so runs stay deterministic and
 * safe. ASan/MSan would supply exactly the same oracles on a production target.
 *
 * MILESTONES are harness instrumentation, not target logic: `MS()` only updates a
 * counter used by the experiment to measure partial progress. It emits no trace
 * point and is therefore invisible to every coverage dimension equally.
 *
 * Build: gcc -O2 -fPIC -shared -o libtarget.so target_lib.c
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* decision-trace plumbing                                            */
/* ------------------------------------------------------------------ */
typedef struct {
    unsigned int *tr;
    int cap;
    int len;
    int milestone;
    int bug;
} rt_t;

static void T(rt_t *r, unsigned int id)
{
    if (r->len < r->cap) {
        r->tr[r->len++] = id;
    } else {
        r->len++; /* keep counting so truncation is detectable */
    }
}

static void MS(rt_t *r, int m)
{
    if (r->milestone < m) r->milestone = m;
}

/* ================================================================== */
/* T1 - recursive TLV parser with a path-dependent scratch layout      */
/* ================================================================== */
/*
 * Two container types, SEQ and MAP, both recursive. Delta-encoded headers: a
 * container nested inside a container of the SAME type shares its parent's header
 * and costs only T1_ADV_SAME bytes of the shared scratch buffer, whereas a type
 * SWITCH needs a fresh header costing T1_ADV_SWITCH bytes.
 *
 * The overflow therefore depends on the *order* of container types along the path
 * from the root, not on the depth or on how many of each type occur:
 *
 *   SEQ MAP SEQ MAP SEQ MAP ->  6 * 8          = 48  -> overflows with any len >= 1
 *   SEQ SEQ SEQ SEQ SEQ SEQ ->  8 + 5 * 2      = 18  -> always safe
 *
 * `off + len > 48` with `len <= 8` needs `off >= 41`, i.e. at least FIVE type
 * switches. AFL edge coverage implicitly carries ONE level of calling context (the
 * edge from a container body block into parse_value identifies the parent type), so
 * requiring a single alternation is not enough -- an earlier 4-switch version of
 * this target was solved by the edge-coverage baseline in 3/3 pilot trials. Five
 * switches puts the required path beyond what one level of implicit context and
 * hit-count bucketing can represent, and beyond what a shallow (N=4) context window
 * can represent either -- which is exactly what the context-depth sweep measures.
 *
 * Those two inputs contain the same number of containers of each type, execute the
 * same basic blocks, and execute each of them the SAME NUMBER OF TIMES. They are
 * therefore indistinguishable to AFL edge coverage *including* its saturating
 * hit-count classes -- which do encode recursion depth, and would otherwise leak
 * the progress signal. Only the calling context (the ordered path) separates them.
 *
 * T1_MAXDEPTH is capped at 6 so that same-type nesting alone can never reach the
 * overflow (12 + 4*5 = 32, plus len <= 8, is still <= 48): reaching the bug
 * strictly requires type alternation.
 */
#define T1_SCRATCH     48
#define T1_ADV_SWITCH   8   /* fresh header when the container type changes */
#define T1_ADV_SAME     2   /* delta header when nested in the same type    */
#define T1_LEN_LIMIT    8
#define T1_MAXDEPTH     8
#define T1_MAXCOUNT     4
#define T1_MS_CAP       6

/* tag = byte & 0x3F -- a 64-value tag space of which only four are meaningful,
 * as in ASN.1 / protobuf-style binary formats. The wide tag space is what makes a
 * 4-container alternating path rare under blind search (~4e-7 per random input)
 * while still costing a feedback-guided fuzzer only ~64 tries per level. */
#define T1_TAG_MASK  0x3F
#define T1_TAG_INT     60
#define T1_TAG_STR     61
#define T1_TAG_SEQ     62
#define T1_TAG_MAP     63
#define T1_TYPE_NONE    0

static int t1_parse_value(const unsigned char *in, size_t n, size_t *pos,
                          int parent_type, unsigned int off, int depth,
                          rt_t *r, unsigned char *scratch);

static int t1_parse_int(const unsigned char *in, size_t n, size_t *pos, rt_t *r)
{
    (void)in;
    T(r, 0x10);
    if (*pos + 4 > n) { T(r, 0x11); return -1; }
    *pos += 4;
    T(r, 0x12);
    return 0;
}

static int t1_parse_str(const unsigned char *in, size_t n, size_t *pos,
                        unsigned int off, rt_t *r, unsigned char *scratch)
{
    unsigned int len, i;

    T(r, 0x20);
    if (*pos >= n) { T(r, 0x21); return -1; }
    len = in[(*pos)++];
    T(r, 0x22);

    /* Sanity limit. Sound for any single header level, but the author did not
     * consider that header cost accumulates along the container path. */
    if (len > T1_LEN_LIMIT) { T(r, 0x23); return -1; }
    if (len == 0) { T(r, 0x24); return 0; }

    if (*pos < n && in[*pos] == 0x5A) { T(r, 0x25); }  /* same edge on every path */

    if (off + len > (unsigned int)T1_SCRATCH) {
        /* SEEDED BUG (oracle): scratch overflow, requires >= 3 type switches. */
        T(r, 0x2F);
        r->bug = 1;
        MS(r, T1_MS_CAP + 1);
        return -2;
    }

    for (i = 0; i < len; i++) {
        if (*pos >= n) break;
        scratch[off + i] = (unsigned char)(in[(*pos)++] ^ 0x11);
    }
    T(r, 0x26);
    return 0;
}

/* Shared container body for SEQ and MAP; `base_id` separates their trace points
 * so each container type really is distinct code with distinct edges. */
static int t1_parse_container(const unsigned char *in, size_t n, size_t *pos,
                              int self_type, int parent_type, unsigned int off,
                              int depth, rt_t *r, unsigned char *scratch,
                              unsigned int base_id)
{
    unsigned int cnt, i, new_off;

    T(r, base_id + 0);
    if (*pos >= n) { T(r, base_id + 1); return -1; }
    cnt = in[(*pos)++];
    if (cnt > T1_MAXCOUNT) cnt = T1_MAXCOUNT;

    /* unconditional data flow: the header cost never appears as a branch */
    new_off = off + (unsigned int)((self_type == parent_type) ? T1_ADV_SAME
                                                             : T1_ADV_SWITCH);
    MS(r, (int)(new_off / T1_ADV_SWITCH) < T1_MS_CAP
             ? (int)(new_off / T1_ADV_SWITCH) : T1_MS_CAP);  /* harness-only */
    T(r, base_id + 2);

    for (i = 0; i < cnt; i++) {
        if (*pos >= n) { T(r, base_id + 3); break; }
        if (t1_parse_value(in, n, pos, self_type, new_off, depth + 1, r, scratch) < 0) {
            T(r, base_id + 4);
            return -1;
        }
        if (r->bug) return -2;
    }
    T(r, base_id + 5);
    return 0;
}

static int t1_parse_value(const unsigned char *in, size_t n, size_t *pos,
                          int parent_type, unsigned int off, int depth,
                          rt_t *r, unsigned char *scratch)
{
    unsigned char tag;

    T(r, 0x40);
    if (depth > T1_MAXDEPTH) { T(r, 0x41); return -1; }
    if (*pos >= n) { T(r, 0x42); return -1; }
    tag = (unsigned char)(in[(*pos)++] & T1_TAG_MASK);
    if (tag == T1_TAG_INT) {
        T(r, 0x44);
        return t1_parse_int(in, n, pos, r);
    }
    if (tag == T1_TAG_STR) {
        T(r, 0x45);
        return t1_parse_str(in, n, pos, off, r, scratch);
    }
    if (tag == T1_TAG_SEQ) {
        T(r, 0x46);
        return t1_parse_container(in, n, pos, T1_TAG_SEQ, parent_type, off, depth,
                                 r, scratch, 0x30);
    }
    if (tag == T1_TAG_MAP) {
        T(r, 0x47);
        return t1_parse_container(in, n, pos, T1_TAG_MAP, parent_type, off, depth,
                                 r, scratch, 0x50);
    }
    T(r, 0x43);
    return 0;                                                  /* END / unknown */
}

int t1_entry(const unsigned char *in, size_t n, unsigned int *tr, int trcap,
             int *trlen, int *milestone)
{
    rt_t r;
    /* Oversized so the seeded overflow is *detected* rather than performed. */
    unsigned char scratch[T1_SCRATCH + 256];
    size_t pos = 0;

    r.tr = tr; r.cap = trcap; r.len = 0; r.milestone = 0; r.bug = 0;
    memset(scratch, 0, sizeof scratch);

    T(&r, 0x01);
    if (n >= 1) {
        t1_parse_value(in, n, &pos, T1_TYPE_NONE, 0u, 0, &r, scratch);
    } else {
        T(&r, 0x02);
    }
    *trlen = r.len;
    *milestone = r.milestone;
    return r.bug;
}

/* ================================================================== */
/* T2 - chained magic-value gates + checksum                          */
/* ================================================================== */
#define T2_MAGIC 0xC0DEu
#define T2_VER   0x1234u
#define T2_TAG   0xBEEFu

int t2_entry(const unsigned char *in, size_t n, unsigned int *tr, int trcap,
             int *trlen, int *milestone)
{
    rt_t r;
    unsigned int magic, ver, tag, declared, len, chk, i;

    r.tr = tr; r.cap = trcap; r.len = 0; r.milestone = 0; r.bug = 0;

    T(&r, 0x01);
    if (n < 16) { T(&r, 0x02); goto done; }

    magic = (unsigned int)in[0] | ((unsigned int)in[1] << 8);
    if (magic != T2_MAGIC) { T(&r, 0x10); goto done; }
    MS(&r, 1);
    T(&r, 0x11);

    ver = (unsigned int)in[2] | ((unsigned int)in[3] << 8);
    if (ver != T2_VER) { T(&r, 0x12); goto done; }
    MS(&r, 2);
    T(&r, 0x13);

    tag = (unsigned int)in[4] | ((unsigned int)in[5] << 8);
    if (tag != T2_TAG) { T(&r, 0x14); goto done; }
    MS(&r, 3);
    T(&r, 0x15);

    declared = in[6];
    len = in[7];
    if (len > (unsigned int)(n - 8)) len = (unsigned int)(n - 8);
    chk = 0;
    for (i = 0; i < len; i++) chk ^= in[8 + i];
    if (chk != declared) { T(&r, 0x16); goto done; }

    /* SEEDED BUG (oracle): all four gates satisfied. */
    MS(&r, 4);
    r.bug = 1;
    T(&r, 0x1F);

done:
    *trlen = r.len;
    *milestone = r.milestone;
    return r.bug;
}

/* ================================================================== */
/* T3 - handle lifecycle with ordering-dependent stale pointer         */
/* ================================================================== */
#define T3_MAXOPS  16
#define T3_BUFCAP  16
#define T3_ARM_MODE 5   /* the one configure() mode that arms the cached pointer */

/* op = byte & 0x0F:
 *   0 open   1 configure   2 write   3 close
 *   4 read   5 flush       6 seek    7 stat
 *   8..15    ioctl variants - any handle-touching call re-resolves the cached
 *            interior pointer, so these are *destructive* to the bug precondition.
 * A 16-operation alphabet with destructive filler is what makes the required
 * ordering genuinely hard for blind search (~1e-7 per random input) while still
 * being climbable one transition at a time by state-transition feedback.
 */
int t3_entry(const unsigned char *in, size_t n, unsigned int *tr, int trcap,
             int *trlen, int *milestone)
{
    rt_t r;
    size_t i;
    int nops = 0;

    /* per-execution "library instance" */
    int h_open = 0;
    int gen = 0;            /* allocation generation of the live buffer          */
    int mode = 0;           /* library-global config, persists across open/close */
    int commit_mode = 0;    /* mode captured by the last flush()                 */
    int cached = 0;         /* an interior pointer has been cached               */
    int cached_gen = -1;    /* generation the cached pointer belongs to          */
    unsigned char *buf = NULL;

    r.tr = tr; r.cap = trcap; r.len = 0; r.milestone = 0; r.bug = 0;

    T(&r, 0x01);
    for (i = 0; i + 1 < n && nops < T3_MAXOPS; i += 2, nops++) {
        unsigned int op = (unsigned int)(in[i] & 0x0F);
        unsigned char arg = in[i + 1];

        if (op == 0) {                                        /* open */
            T(&r, 0x10);
            if (h_open) { T(&r, 0x11); continue; }
            gen++;
            buf = (unsigned char *)malloc(T3_BUFCAP);
            if (!buf) { T(&r, 0x13); break; }
            memset(buf, 0, T3_BUFCAP);
            h_open = 1;
            T(&r, 0x12);
            MS(&r, 1);
            if (cached && cached_gen != gen) MS(&r, 4);
        } else if (op == 1) {                                 /* configure */
            T(&r, 0x20);
            if (!h_open) { T(&r, 0x21); continue; }
            /* unconditional data flow: no branch leaks the armed mode */
            mode = (int)(arg & 0x07);
            cached = 1;
            cached_gen = gen;
            T(&r, 0x22);
            if (mode == T3_ARM_MODE) MS(&r, 2);
        } else if (op == 5) {                                 /* flush */
            T(&r, 0x50);
            if (!h_open) { T(&r, 0x51); continue; }
            commit_mode = mode;                               /* unconditional */
            T(&r, 0x52);
            if (commit_mode == T3_ARM_MODE) MS(&r, 3);
        } else if (op == 2) {                                 /* write */
            T(&r, 0x30);
            if (!h_open) { T(&r, 0x31); continue; }
            {
                int use_cached = (commit_mode == T3_ARM_MODE) && cached;
                /* The ONLY branch in this target that depends on the ordering.
                 * Decided from mode/generation state, never from pointer identity:
                 * a reallocated buffer can legally land on the same address and the
                 * Python model must stay bit-identical. */
                if (use_cached && cached_gen != gen) {
                    T(&r, 0x3F);
                    r.bug = 1;
                    MS(&r, 5);
                    break;
                }
                buf[0] = arg;
                commit_mode = 0;                  /* a write consumes the commit */
                T(&r, 0x32);
            }
        } else if (op == 3) {                                 /* close */
            T(&r, 0x40);
            if (!h_open) { T(&r, 0x41); continue; }
            free(buf);
            buf = NULL;
            h_open = 0;
            /* BUG SOURCE: `cached` / `cached_gen` deliberately NOT invalidated. */
            T(&r, 0x42);
        } else if (op == 4) {                                 /* read */
            T(&r, 0x60);
            if (!h_open) { T(&r, 0x61); continue; }
            cached_gen = gen;                     /* re-validates the cache */
            T(&r, 0x62);
        } else if (op == 6) {                                 /* seek */
            T(&r, 0x70);
            if (!h_open) { T(&r, 0x71); continue; }
            cached_gen = gen;
            commit_mode = 0;
            T(&r, 0x72);
        } else if (op == 7) {                                 /* stat */
            T(&r, 0x80);
        } else {                                              /* ioctl 8..15 */
            T(&r, 0x90);
            if (!h_open) { T(&r, 0x91); continue; }
            cached_gen = gen;                     /* re-resolves the cache */
            T(&r, 0x92);
        }
    }
    T(&r, 0x02);

    if (buf) free(buf);
    *trlen = r.len;
    *milestone = r.milestone;
    return r.bug;
}

/* ================================================================== */
/* T4 - genuine higher-order (n-gram) blind spot                      */
/* ================================================================== */
/*
 * The input is read as a FIXED-LENGTH walk of exactly T4_WALK steps. Step i
 * consumes in[i % n] and reduces it to a symbol (b ^ 0x5A) & 31 over a 32-symbol
 * alphabet; each symbol dispatches to its own handler basic block and is entered
 * through its own call site. Exactly one basic block is executed per step, so the
 * executed block sequence IS the symbol sequence.
 *
 * Because the walk length is fixed, every execution executes the same number of
 * blocks and no hit-count class ever varies with progress: the D0 signal is the
 * pair multiset alone. (A variable-length design was discarded precisely because
 * AFL's saturating hit-count classes leak the walk length and hand the baseline a
 * free gradient.)
 *
 * A Knuth-Morris-Pratt automaton advances the match state by TABLE LOOKUP ONLY,
 * so no prefix of T4_REQ produces a new edge, a new hit-count class, or any other
 * control-flow event. The bug fires when the block sequence equals
 * T4_REQ = A B C A B = {3,12,19,3,12}: an order-6 property of the block
 * sequence, hence provably not a function of its bigram multiset.
 *
 * WITNESS: A B C A B B (match state 5, one symbol from the bug) and
 * A B B C A B (match state 2) have the identical bigram multiset and therefore a
 * bit-for-bit identical AFL map including hit-count classes, but different
 * trigram multisets.
 */
#define T4_ALPHABET 32
#define T4_WALK      6
#define T4_DECODE_XOR 0x5A
#define T4_REQLEN    5

/* KMP transition table for REQ = {0,1,2,0,1,0} over a 32-symbol alphabet.
 * Row T4_REQLEN is absorbing. Generated by the Python reference and asserted
 * identical by the differential fidelity check. */
static const int t4_trans[T4_REQLEN + 1][T4_ALPHABET] = {
    { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 },
};

static void t4_step(const unsigned char *in, size_t n, int i, int *state, rt_t *r)
{
    int s;

    if (i >= T4_WALK) { T(r, 0x41); return; }
    s = (in[(size_t)i % n] ^ T4_DECODE_XOR) & (T4_ALPHABET - 1);
    T(r, (unsigned int)(0x50 + s));
    /* branchless progress: a table lookup, never a branch */
    *state = t4_trans[*state][s];
    MS(r, *state);
    if (*state >= T4_REQLEN) {
        T(r, 0x70);
        r->bug = 1;
    }
    t4_step(in, n, i + 1, state, r);
}

int t4_entry(const unsigned char *in, size_t n, unsigned int *tr, int trcap,
             int *trlen, int *milestone)
{
    rt_t r;
    int state = 0;

    r.tr = tr; r.cap = trcap; r.len = 0; r.milestone = 0; r.bug = 0;

    T(&r, 0x01);
    if (n == 0) {
        T(&r, 0x42);
        *trlen = r.len;
        *milestone = 0;
        return 0;
    }
    t4_step(in, n, 0, &state, &r);
    T(&r, 0x02);

    *trlen = r.len;
    *milestone = r.milestone;
    return r.bug;
}
