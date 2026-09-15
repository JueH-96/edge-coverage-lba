#!/usr/bin/env bash
#
# One-click artifact evaluation.
#
#   ./run_ae.sh --quick          full verification from bundled data   (< 2 minutes)
#   ./run_ae.sh --full           re-run the fuzzing campaigns from scratch (hours)
#   ./run_ae.sh --full --workers 16 --trials 30
#
# --quick is the mode reviewers should run first. It is not a reduced or smoke
# version of the evaluation: it builds the native libraries, runs the complete
# 75-test invariant suite, performs all 20,000 native-vs-Python fidelity
# comparisons, regenerates the scheduler analysis from the bundled raw trial
# records, and redraws every figure. The only thing it does *not* do is re-execute
# the multi-hour fuzzing campaigns that produced those trial records -- that is
# what --full is for.
#
set -euo pipefail

AE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AE_ROOT"

MODE=""
WORKERS=12
TRIALS=30
EXECS=50000
BUDGET=150000

usage() {
    sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --quick             verify everything from the bundled trial records (default)
  --full              additionally re-run the fuzzing campaigns from scratch
  --workers N         parallel worker processes for --full   (default 12)
  --trials N          trials per cell for --full             (default 30)
  --execs N           execs per trial, scheduler matrix      (default 50000)
  --budget N          execs per trial, order sweep           (default 150000)
  -h, --help          show this help

Exit status is 0 only if every stage passed.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)   MODE="quick"; shift ;;
        --full)    MODE="full";  shift ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --trials)  TRIALS="$2";  shift 2 ;;
        --execs)   EXECS="$2";   shift 2 ;;
        --budget)  BUDGET="$2";  shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done
MODE="${MODE:-quick}"

# --------------------------------------------------------------------------- #
# Interpreter selection.
#
# Rather than guessing, try each candidate and keep the first one that can
# actually import the four dependencies. This means the artifact works whether
# the reviewer used a venv, uv, conda, or a system Python with the packages
# already installed -- and it fails with a clear message rather than a traceback
# if none of them can.
# --------------------------------------------------------------------------- #
DEPS_PROBE='import numpy, scipy, matplotlib, pytest'

py_works() { "$@" -c "$DEPS_PROBE" >/dev/null 2>&1; }

PY=()
if [[ -n "${AE_PYTHON:-}" ]]; then
    # Explicit override: honour it, but say so loudly if it is unusable.
    PY=("$AE_PYTHON")
    if ! py_works "${PY[@]}"; then
        echo "ERROR: AE_PYTHON=$AE_PYTHON cannot import numpy/scipy/matplotlib/pytest" >&2
        exit 1
    fi
else
    if [[ -x "$AE_ROOT/.venv/bin/python" ]] && py_works "$AE_ROOT/.venv/bin/python"; then
        PY=("$AE_ROOT/.venv/bin/python")
    elif py_works python3; then
        PY=(python3)
    elif command -v uv >/dev/null 2>&1 && py_works uv run python; then
        PY=(uv run python)
    elif py_works python; then
        PY=(python)
    fi
fi

if [[ ${#PY[@]} -eq 0 ]]; then
    cat >&2 <<'EOF'
ERROR: no usable Python interpreter found.

The artifact needs numpy, scipy, matplotlib and pytest. Install them with any of:

    python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
    pip install -r requirements.txt
    uv sync

or point the artifact at an existing interpreter:

    AE_PYTHON=/path/to/python ./run_ae.sh --quick

See INSTALL.md for details.
EOF
    exit 1
fi

STAGE_LOG="$AE_ROOT/results/run_ae_stages.log"
mkdir -p "$AE_ROOT/results" "$AE_ROOT/figures"
: > "$STAGE_LOG"

FAILED=0
START_ALL=$(date +%s)

banner() {
    echo ""
    echo "==============================================================="
    echo "  $1"
    echo "==============================================================="
}

# stage <name> <command...>
stage() {
    local name="$1"; shift
    local t0 t1 rc
    banner "$name"
    t0=$(date +%s)
    set +e
    "$@"
    rc=$?
    set -e
    t1=$(date +%s)
    if [[ $rc -eq 0 ]]; then
        echo "PASS  $name  ($((t1 - t0))s)" | tee -a "$STAGE_LOG"
    else
        echo "FAIL  $name  (exit $rc, $((t1 - t0))s)" | tee -a "$STAGE_LOG"
        FAILED=1
    fi
}

echo "artifact root : $AE_ROOT"
echo "mode          : $MODE"
echo "interpreter   : ${PY[*]}"
"${PY[@]}" -c "import sys; print('python        :', sys.version.split()[0])"

# --------------------------------------------------------------------------- #
# Stage 1 - build the native C ground-truth libraries
# --------------------------------------------------------------------------- #
stage "1/5  Build native benchmark libraries" \
    "${PY[@]}" experiments/build_native.py

# --------------------------------------------------------------------------- #
# Stage 2 - 75 invariant tests (theory + implementation)
# --------------------------------------------------------------------------- #
stage "2/5  Invariant suite (75 tests)" \
    "${PY[@]}" -m pytest tests/test_invariants.py -q

# --------------------------------------------------------------------------- #
# Stage 3 - 20,000 native-C vs Python fidelity comparisons
# --------------------------------------------------------------------------- #
stage "3/5  Fidelity cross-validation (20,000 comparisons)" \
    "${PY[@]}" tests/test_fidelity.py

# --------------------------------------------------------------------------- #
# Stage 3b (--full only) - re-run the campaigns that produced data/*.json
# --------------------------------------------------------------------------- #
if [[ "$MODE" == "full" ]]; then
    banner "FULL MODE: re-running the fuzzing campaigns from scratch"
    echo "This overwrites the bundled trial records in data/ and takes hours."
    echo "workers=$WORKERS trials=$TRIALS execs=$EXECS budget=$BUDGET"

    stage "3b.1  Scheduler campaign matrix (3 targets x 9 arms x $TRIALS trials)" \
        "${PY[@]}" experiments/run_scheduler_matrix.py \
            --trials "$TRIALS" --execs "$EXECS" --workers "$WORKERS"

    stage "3b.2  T4 n-gram order sweep (NGRAM 2-8)" \
        "${PY[@]}" experiments/run_order_sweep.py \
            --trials "$TRIALS" --budget "$BUDGET" --workers "$WORKERS"

    stage "3b.3  Fixed-capacity corpus dilution ablation (K=1000)" \
        "${PY[@]}" experiments/run_corpus_ablation.py \
            --trials "$TRIALS" --workers "$WORKERS"
fi

# --------------------------------------------------------------------------- #
# Stage 4 - analysis: statistics, FDR family, summary tables
# --------------------------------------------------------------------------- #
stage "4/5  Analysis, statistics and summary tables" \
    "${PY[@]}" experiments/analyze_results.py

# --------------------------------------------------------------------------- #
# Stage 5 - figures
# --------------------------------------------------------------------------- #
stage "5/5  Regenerate all publication figures" \
    "${PY[@]}" plots/generate_all_figures.py

# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
END_ALL=$(date +%s)
banner "ARTIFACT EVALUATION SUMMARY"
cat "$STAGE_LOG"
echo ""
echo "total wall time : $((END_ALL - START_ALL))s"
echo "results         : $AE_ROOT/results"
echo "figures         : $AE_ROOT/figures  ($(find figures -name '*.pdf' | wc -l) PDF, $(find figures -name '*.png' | wc -l) PNG; figures/paper/ holds the submission figures)"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo "RESULT: ALL STAGES PASSED"
    echo ""
    echo "Read results/summary_tables.md for the claim-by-claim tables and"
    echo "results/ae_summary.json for the machine-readable summary (including,"
    echo "for each result document, whether it was regenerated here or bundled)."
    exit 0
else
    echo "RESULT: ONE OR MORE STAGES FAILED (see the log above)"
    exit 1
fi
