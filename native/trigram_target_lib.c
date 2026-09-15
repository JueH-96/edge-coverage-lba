/*
 * trigram_target_lib.c - native ground truth for the Step 2 trigram benchmark.
 *
 * Three targets whose triggering predicates are properties of the executed
 * basic-block sequence of order three or higher, and therefore - by the Step 1
 * characterisation of an AFL map as a function of the block *bigram multiset* -
 * provably invisible to edge coverage however large the map and however finely
 * its hit counts are classified.
 *
 * The Python models in ../trigram_targets.py mirror this file statement for
 * statement. check_trigram_fidelity() pushes 5,000 random and near-solution
 * inputs through both and compares the decision trace, milestone and bug flag
 * exactly; this file is the ground truth in that comparison, which is what lets
 * the experiments run against the Python models without the results being a
 * claim about Python.
 *
 * Three invariants are maintained on purpose and are easy to break by accident:
 *
 *  1. FIXED WALK LENGTH. Every execution of a walk target emits exactly the same
 *     number of trace points on the non-triggering path, so AFL's saturating
 *     hit-count classes never leak progress. A variable-length walk would hand
 *     the edge-coverage baseline a free gradient and invalidate the comparison.
 *
 *  2. BRANCHLESS PROGRESS. Advancing toward the bug is a table lookup or an
 *     arithmetic select, never an `if`. A branch on partial progress would emit
 *     a new edge and destroy the blind spot the benchmark exists to exhibit.
 *
 *  3. MILESTONES ARE NOT TRACE POINTS. MS() only raises a counter the harness
 *     reads to measure partial progress; it emits nothing, so it is invisible to
 *     every coverage dimension equally and grants no configuration an advantage.
 *
 * Seeded bugs use explicit oracles rather than real memory corruption, so runs
 * stay deterministic and safe; ASan would supply the same oracles on a
 * production target.
 *
 * Build: gcc -O2 -fPIC -shared -o libtrigram_target.so trigram_target_lib.c
 */

#include <stddef.h>
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
        r->len++; /* keep counting so truncation stays detectable */
    }
}

static void MS(rt_t *r, int m)
{
    if (r->milestone < m) r->milestone = m;
}

/* ================================================================== */
/* TG1 - pure order-3 trigram lock                                     */
/* ================================================================== */
/*
 * The bug needs the first six overlapping trigrams of the key word A B A C A B
 * C B A C, IN ORDER at increasing walk positions, over symbols A = 6, B = 1, C = 4
 * of an 8-symbol alphabet. A long shallow ladder, not a short steep one: each
 * rung costs about one byte, and it is having EIGHT of them in sequence that is
 * hard, which makes the benchmark a test of corpus retention rather than luck. Advancing is `prog += advtab[prog][trigram]`, a table lookup with no
 * branch, so no rung of the ladder is observable in control flow.
 *
 * The four constituent BIGRAMS {AB x2, BA, AC, CA} are individually cheap, so an
 * edge-coverage fuzzer collects them early and then has no gradient left: the
 * walk A C A B A B is a different Eulerian trail of the same transition
 * multigraph, has an identical bigram multiset and identical per-symbol counts -
 * hence a bit-for-bit identical AFL map, count classes included - yet it never
 * gets past the first rung.
 */
#define TG1_ALPHABET  8
#define TG1_MASK     (TG1_ALPHABET - 1)
#define TG1_XOR      0x5A
#define TG1_WALK     16
#define TG1_A         6
#define TG1_B         1
#define TG1_C         4
#define TG1_NREQ      6

/* Fixed 4-bit-per-symbol packing, independent of the alphabet size. Keeping the
 * packing width fixed while the alphabet is retuned costs a few unused kilobytes
 * and removes a whole class of silent C/Python index drift. */
#define TRI_TAB_SIZE (16 * 16 * 16)
#define TRI_ID(a, b, c) (((unsigned)(a) << 8) | ((unsigned)(b) << 4) | (unsigned)(c))

/* Ordered-ladder advance table: prog += advtab[prog][tri], which is 1 exactly
 * when tri is the trigram the ladder currently waits for. Row TG1_NREQ is all
 * zeros, so the top of the ladder is absorbing. An earlier design required the
 * trigrams as an unordered SET; a pilot showed that closing the last rung then
 * demanded global consistency across the whole walk rather than one more byte,
 * so every condition stalled one rung short and the benchmark discriminated
 * nothing. In order, each rung is reachable from the one below by a local
 * mutation, and the difference between conditions becomes whether the feedback
 * RETAINS the intermediate seed - which is the question under study. */
static unsigned char tg1_advtab[TG1_NREQ + 1][TRI_TAB_SIZE];
static unsigned char tg1_ready = 0;

static void tg1_init(void)
{
    if (tg1_ready) return;
    memset(tg1_advtab, 0, sizeof(tg1_advtab));
    /* overlapping trigrams of the key word A B A C A B C B A C */
    tg1_advtab[0][TRI_ID(TG1_A, TG1_B, TG1_A)] = 1;
    tg1_advtab[1][TRI_ID(TG1_B, TG1_A, TG1_C)] = 1;
    tg1_advtab[2][TRI_ID(TG1_A, TG1_C, TG1_A)] = 1;
    tg1_advtab[3][TRI_ID(TG1_C, TG1_A, TG1_B)] = 1;
    tg1_advtab[4][TRI_ID(TG1_A, TG1_B, TG1_C)] = 1;
    tg1_advtab[5][TRI_ID(TG1_B, TG1_C, TG1_B)] = 1;
    tg1_ready = 1;
}

/* The recursion IS the walk: exactly one handler block per symbol, so the
 * executed block sequence is the symbol sequence verbatim and a D1 context
 * window of depth N is the trailing symbol N-gram. */
static void tg1_step(const unsigned char *in, size_t n, int i,
                     int *prog, int *p1, int *p2, rt_t *r)
{
    unsigned int s, tri;

    if (i >= TG1_WALK) {
        T(r, 0x41);
        return;
    }
    s = (unsigned int)((in[(size_t)i % n] ^ TG1_XOR) & TG1_MASK);
    T(r, 0x50u + s);

    if (i >= 2) {
        tri = TRI_ID(*p2, *p1, s);
        *prog += (int)tg1_advtab[*prog][tri];
    }
    *p2 = *p1;
    *p1 = (int)s;
    MS(r, *prog);

    tg1_step(in, n, i + 1, prog, p1, p2, r);
}

int tg1_entry(const unsigned char *in, size_t n, unsigned int *tr, int trcap,
              int *trlen, int *milestone)
{
    rt_t r;
    int prog = 0;
    int p1 = 0, p2 = 0;

    tg1_init();
    r.tr = tr; r.cap = trcap; r.len = 0; r.milestone = 0; r.bug = 0;

    T(&r, 0x01);
    if (n == 0) {
        T(&r, 0x42);
        *trlen = r.len;
        *milestone = 0;
        return 0;
    }
    tg1_step(in, n, 0, &prog, &p1, &p2, &r);
    if (prog >= TG1_NREQ) {
        T(&r, 0x70);
        r.bug = 1;
    }
    T(&r, 0x02);

    *trlen = r.len;
    *milestone = r.milestone;
    return r.bug;
}

/* ================================================================== */
/* TG2 - order-3 gate unlocking a 32-bit value gate                    */
/* ================================================================== */
/*
 * Phase one is an ordered two-trigram ladder (key word A B A C). Phase two compares a
 * little-endian 32-bit word at a fixed offset against a magic constant. The
 * comparison site is UNREACHABLE until phase one completes, so the two
 * objectives are strictly sequential: a value-range-only fuzzer never gets to
 * use its byte-wise gradient, and a context-only fuzzer stalls on 2^32.
 *
 * The per-byte match count feeds the milestone but emits no trace point, so it
 * grants no configuration an edge signal.
 */
#define TG2_ALPHABET  8
#define TG2_MASK     (TG2_ALPHABET - 1)
#define TG2_XOR      0x5A
#define TG2_WALK     10
#define TG2_A         6
#define TG2_B         1
#define TG2_C         4
#define TG2_NREQ      3
#define TG2_VAL_OFF   8
#define TG2_MAGIC    0x5AC31E7Bu

static unsigned char tg2_advtab[TG2_NREQ + 1][TRI_TAB_SIZE];
static unsigned char tg2_ready = 0;

static void tg2_init(void)
{
    if (tg2_ready) return;
    memset(tg2_advtab, 0, sizeof(tg2_advtab));
    /* overlapping trigrams of the key word A B A C A B */
    tg2_advtab[0][TRI_ID(TG2_A, TG2_B, TG2_A)] = 1;
    tg2_advtab[1][TRI_ID(TG2_B, TG2_A, TG2_C)] = 1;
    tg2_advtab[2][TRI_ID(TG2_A, TG2_C, TG2_A)] = 1;
    tg2_ready = 1;
}

static void tg2_step(const unsigned char *in, size_t n, int i,
                     int *prog, int *p1, int *p2, rt_t *r)
{
    unsigned int s, tri;

    if (i >= TG2_WALK) {
        T(r, 0x41);
        return;
    }
    s = (unsigned int)((in[(size_t)i % n] ^ TG2_XOR) & TG2_MASK);
    T(r, 0x50u + s);

    if (i >= 2) {
        tri = TRI_ID(*p2, *p1, s);
        *prog += (int)tg2_advtab[*prog][tri];
    }
    *p2 = *p1;
    *p1 = (int)s;
    MS(r, *prog);

    tg2_step(in, n, i + 1, prog, p1, p2, r);
}

int tg2_entry(const unsigned char *in, size_t n, unsigned int *tr, int trcap,
              int *trlen, int *milestone)
{
    rt_t r;
    int prog = 0;
    int p1 = 0, p2 = 0;
    unsigned int v;
    int k, nbytes;

    tg2_init();
    r.tr = tr; r.cap = trcap; r.len = 0; r.milestone = 0; r.bug = 0;

    T(&r, 0x01);
    if (n == 0) {
        T(&r, 0x42);
        *trlen = r.len;
        *milestone = 0;
        return 0;
    }
    tg2_step(in, n, 0, &prog, &p1, &p2, &r);

    if (prog < TG2_NREQ) {
        T(&r, 0x43);            /* phase one incomplete: comparison unreachable */
        T(&r, 0x02);
        *trlen = r.len;
        *milestone = r.milestone;
        return 0;
    }
    T(&r, 0x44);                /* phase two entered */

    if (n < (size_t)TG2_VAL_OFF + 4u) {
        T(&r, 0x45);            /* operand truncated */
        T(&r, 0x02);
        *trlen = r.len;
        *milestone = r.milestone;
        return 0;
    }

    v = (unsigned int)in[TG2_VAL_OFF]
      | ((unsigned int)in[TG2_VAL_OFF + 1] << 8)
      | ((unsigned int)in[TG2_VAL_OFF + 2] << 16)
      | ((unsigned int)in[TG2_VAL_OFF + 3] << 24);

    nbytes = 0;
    for (k = 0; k < 4; k++) {
        nbytes += (int)(((v >> (8 * k)) & 0xFFu) == ((TG2_MAGIC >> (8 * k)) & 0xFFu));
    }
    MS(&r, TG2_NREQ + nbytes);

    if (v == TG2_MAGIC) {
        T(&r, 0x70);
        r.bug = 1;
    }
    T(&r, 0x02);

    *trlen = r.len;
    *milestone = r.milestone;
    return r.bug;
}

/* ================================================================== */
/* TG3 - rare-transition chain, compact under a state abstraction      */
/* ================================================================== */
/*
 * TG3_NOPS decoded ops drive a TG3_NSTATES-state machine. From state s the
 * single op tg3_advance[s] moves to s + 1; every other op returns to state 0.
 * The final state is absorbing, so the bug latches once reached.
 *
 * The state update is an arithmetic select rather than a branch, so partial
 * progress along the chain is invisible in control flow. The op handlers DO emit
 * one block each, so edge coverage and the context window see the op sequence -
 * this target is a fair fight, and the point it makes is about the SIZE of the
 * abstraction that suffices: the chain is a length-6 property, representable by
 * a context window only at depth >= 6 (context space 8^6), and by the state
 * machine in 7 states.
 */
#define TG3_ALPHA    8
#define TG3_OP_MASK  (TG3_ALPHA - 1)
#define TG3_XOR      0x3C
#define TG3_NOPS     12
#define TG3_NSTATES   7

static const int tg3_advance[TG3_NSTATES] = { 6, 3, 5, 1, 6, 2, 0 };

int tg3_entry(const unsigned char *in, size_t n, unsigned int *tr, int trcap,
              int *trlen, int *milestone)
{
    rt_t r;
    int state = 0;
    int i, adv, fin;
    unsigned int op;

    r.tr = tr; r.cap = trcap; r.len = 0; r.milestone = 0; r.bug = 0;

    T(&r, 0x01);
    if (n == 0) {
        T(&r, 0x42);
        *trlen = r.len;
        *milestone = 0;
        return 0;
    }

    for (i = 0; i < TG3_NOPS; i++) {
        op = (unsigned int)((in[(size_t)i % n] ^ TG3_XOR) & TG3_OP_MASK);
        T(&r, 0x50u + op);
        /* branchless advance: arithmetic select, never an `if` */
        adv = (int)((int)op == tg3_advance[state]);
        fin = (int)(state >= TG3_NSTATES - 1);
        state = fin * state + (1 - fin) * (adv * (state + 1));
        MS(&r, state);
    }

    if (state >= TG3_NSTATES - 1) {
        T(&r, 0x70);
        r.bug = 1;
    }
    T(&r, 0x02);

    *trlen = r.len;
    *milestone = r.milestone;
    return r.bug;
}
