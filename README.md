# Multidimensional Guidance for Coverage-Guided Fuzzing

This repository provides a self-contained implementation and evaluation of multidimensional guidance for coverage-guided fuzzing.

It includes invariant tests, fidelity validation, fuzzing experiments, statistical analysis, and reproducible result generation. There are no absolute paths, network dependencies, or external data downloads. The project requires only a C compiler and a numeric Python stack.

## Quick Start

```bash
./run_ae.sh --quick
```

A successful run should report:

```text
PASS  1/5  Build native benchmark libraries
PASS  2/5  Invariant suite (75 tests)
PASS  3/5  Fidelity cross-validation (20,000 comparisons)
PASS  4/5  Analysis, statistics and summary tables
PASS  5/5  Generate result visualisations

RESULT: ALL STAGES PASSED
```

Setup instructions are available in [`INSTALL.md`](INSTALL.md).

---

## 1. Execution Modes

Two execution modes are provided: `--quick` and `--full`.

`--quick` runs the complete invariant suite, fidelity campaign, and statistical analysis using the bundled per-trial records.

`--full` additionally re-executes the fuzzing campaigns from scratch.

| | `--quick` | `--full` |
|---|---|---|
| Build native C libraries | yes | yes |
| 75 invariant tests | yes | yes |
| 20,000 fidelity comparisons | yes | yes |
| Fuzzing campaigns | uses bundled `data/*.json` | re-executed from scratch |
| Statistical analysis and FDR family | recomputed | recomputed |
| Result visualisations | regenerated | regenerated |
| Typical wall time | ~70 s | hours, depending on `--workers` |

Run the full evaluation with:

```bash
./run_ae.sh --full --workers 26 --trials 30
```

The full run overwrites bundled records in `data/`.

Copy the directory first if you want to preserve the original records for comparison.

### Provenance tracking

`results/ae_summary.json` records whether each result document was regenerated during the current run or obtained from the bundled data.

The distinction is determined by comparing content hashes against `data/precomputed/`, preventing data copied by an earlier run from being incorrectly reported as newly generated.

Two documents are bundled under `--quick`:

- `step1_guidance_validation.json`
- `step2_revision.json`

Their analyses are interleaved with fresh sampling and therefore cannot be separated from the multi-hour experiment execution.

Other analyses are recomputed from the raw trial records.

### Recomputation check

`results/step2_scheduler_validation.json` is regenerated from `data/step2_trials.json` on every run.

The regenerated result is compared with the reference copy stored in `data/precomputed/`.

On the reference machine:

```text
[analyze] recomputation vs shipped analysis: reproduces (4168/4168 numeric leaves match)
```

Timing fields are excluded because they depend on the execution environment rather than the analysis itself.

---

## 2. Components and Validation

### Guidance dimensions

The implementation contains four feedback dimensions:

- **D0 — AFL edge coverage**
- **D1 — Calling context**
- **D2 — Value-range information**
- **D3 — Abstract state information**

The corresponding implementations are located in:

```text
src/guidance_engine.py
```

### Core invariant checks

The invariant suite validates properties including:

- factorisation of the feedback signal;
- AFL edge coverage as a bigram statistic over executed block sequences;
- separable information-loss mechanisms;
- refinement relationships between guidance dimensions;
- order-sensitive structural blind spots;
- O(1) rolling prefix hashing against an O(N) reference implementation.

Run the suite with:

```bash
make test
```

or:

```bash
python -m pytest tests/test_invariants.py -v
```

### Main result sources

Important result documents include:

```text
results/step1_guidance_validation.json
results/fidelity_verification.json
results/step2_revision.json
results/step2_scheduler_validation.json
results/corpus_cap_ablation.json
results/summary_tables.md
```

They contain the outputs of the guidance experiments, paired analyses, order sweeps, scheduler experiments, fidelity checks, and corpus-capacity ablations.

---

## 3. Verification

### 75/75 invariant tests

Run:

```bash
make test
```

or:

```bash
python -m pytest tests/test_invariants.py -v
```

`tests/test_invariants.py` aggregates two invariant suites.

#### `_invariants_step1.py`

Contains 22 tests covering:

- guidance engine behaviour;
- refinement relationships;
- O(1) rolling prefix hashing against an O(N) implementation;
- value-range bucketing;
- Miller–Madow entropy estimation;
- Jeffreys-smoothed surprisal;
- AFL bigram characterisation.

#### `_invariants_step2.py`

Contains 53 tests covering:

- energy schedules;
- trigram targets;
- state transitions;
- dilution metrics.

Some tests are parametrised over multiple trigram targets.

### 20,000/20,000 fidelity comparisons

Run:

```bash
make fidelity
```

or:

```bash
python tests/test_fidelity.py
```

The experiments use Python models of the benchmark targets because the guidance probes require access to calling context, value ranges, and abstract state at a finer granularity than the compiled binaries expose.

To validate these models, random inputs are executed against both the Python implementation and the native C implementation.

For each input, the following are compared exactly:

- full decision trace;
- milestone index;
- bug flag.

| Suite | Targets | Inputs per target | Comparisons |
|---|---:|---:|---:|
| Step 1 (T1–T4) | 4 | 2,000 | 8,000 |
| Trigram (TG1–TG3) | 3 | 4,000 | 12,000 |
| **Total** | **7** | | **20,000** |

The trigram suite uses a mixture of uniform-byte inputs and perturbations of the target key word.

This prevents the validation from testing only non-triggering inputs.

`test_fidelity.py` also contains a guard that fails the test if no sampled input passes the first target milestone.

---

## 4. Repository Layout

```text
artifact_evaluation/
├── run_ae.sh
├── Makefile
├── ae_paths.py
├── PORT_LOG.json
│
├── src/
│   ├── guidance_engine.py
│   ├── energy_scheduler.py
│   ├── fuzz_harness.py
│   ├── step2_harness.py
│   ├── stats_utils.py
│   └── stats_paired.py
│
├── native/
│   ├── target_lib.c
│   ├── trigram_target_lib.c
│   └── *.so
│
├── targets/
├── tests/
├── experiments/
├── plots/
├── data/
└── results/
```

### Key modules

#### `src/guidance_engine.py`

Implements:

- `AFLEdgeTracker` (D0)
- `CallingContextTracker` (D1)
- `ValueRangeTracker` (D2)
- `StateMachineTracker` (D3)
- `MultiDimGuidanceEngine`

#### `src/energy_scheduler.py`

Implements:

- round-robin scheduling;
- AFLFast-style scheduling;
- static novelty scheduling;
- adaptive novelty scheduling;
- dynamic depth escalation;
- dilution metrics.

#### `src/fuzz_harness.py`

Provides the AFL-style havoc mutator and blocked-seed trial runner.

#### `src/step2_harness.py`

Provides the trial runner for the trigram experiments.

#### `src/stats_utils.py`

Implements statistical procedures including:

- Vargha–Delaney A12;
- Mann–Whitney tests;
- Wilcoxon tests;
- Clopper–Pearson intervals;
- Fisher's exact test;
- log-rank tests;
- bootstrap confidence intervals;
- Benjamini–Hochberg FDR correction.

#### `src/stats_paired.py`

Contains inference procedures for matched experimental designs.

### Native targets

The `native/` directory contains the native C implementations used as ground truth.

```text
native/target_lib.c
native/trigram_target_lib.c
```

Shared libraries are generated with:

```bash
make build
```

### Tests

The `tests/` directory contains:

```text
test_invariants.py
test_fidelity.py
```

The complete validation consists of:

- 75 invariant tests;
- 20,000 Python-vs-C fidelity comparisons.

### Experiments

The `experiments/` directory contains scripts for:

- native target construction;
- scheduler matrices;
- order sweeps;
- corpus-capacity ablations;
- result analysis.

---

## 5. Bundled Data

The repository includes the raw trial records required to reproduce the statistical analyses.

| File | Contents |
|---|---|
| `data/step1_trials.json` | 540-trial Step 1 bug-finding campaign |
| `data/context_depth_trials.json` | 150-trial context-depth sweep |
| `data/step2_trials.json` | 810-trial scheduler matrix |
| `data/t4_order_sweep.json` | T4 n-gram order sweep under both energy schedules |
| `data/corpus_cap_trials.json` | fixed-capacity dilution ablation |
| `data/precomputed/` | reference copies used for provenance and recomputation checks |

---

## 6. Reproducibility

All randomness is controlled through explicitly seeded `random.Random` and `numpy.random.Generator` instances.

No library-internal or wall-clock entropy is used by the experiments.

### Master seeds

```text
Step 1: 20260804
Step 2: 20260811
```

### Trial seed derivation

Trial seeds are generated using an FNV-1a-style hash of:

```text
(master seed, target name, trial index)
```

The derivation is identical across configurations.

As a result, each experimental arm receives the same seed sequence.

This blocked design reduces differences caused by random seed selection and makes comparisons primarily reflect the feedback mechanism and energy schedule.

### Statistical protocol

The experimental protocol uses:

- at least 30 independent trials per cell;
- explicit right censoring;
- log-rank tests for censored comparisons;
- Vargha–Delaney A12 effect sizes;
- Benjamini–Hochberg FDR correction.

The Step 2 analysis contains 192 hypotheses in its FDR family.

### Fidelity seeds

Seeds used by `tests/test_fidelity.py` are fixed so that the 20,000-comparison fidelity campaign is reproducible.

### Floating-point comparison

Floating-point summation order may vary between BLAS implementations.

For this reason, recomputation checks compare floating-point values using a relative tolerance of:

```text
1e-9
```

Integer counts and Boolean values must match exactly.

---

## 7. Interpreting the Results

Several experiments produce null or weak effects that should be interpreted carefully.

### T1 context sensitivity

For `d0_d1_context`, context sensitivity does not produce a statistically significant time-to-bug advantage at depth N=16:

```text
log-rank p = 0.72
```

### TG1 and TG2 floor effects

TG1 and TG2 contain very few successful bug discoveries:

```text
TG1: at most 1/30 successes in any arm
TG2: 0/30 successes across all nine arms
```

These configurations therefore contain insufficient events for strong comparisons between scheduling strategies.

TG3 contains substantially more successful events and provides more informative comparisons.

### Adaptive energy scheduling

On TG3:

```text
d3d_adaptive_n4: 26/30
d3d_rr_n4:       29/30

log-rank p = 0.785
A12 = 0.443
```

Under the evaluated configuration and budget, the adaptive schedule does not improve over the corresponding round-robin control.

### Limitations

The current evaluation has several limitations:

1. Floor effects on TG1 and TG2 reduce statistical power for the trigram bug endpoint.
2. The adaptive scheduler is evaluated using one parameterisation.
3. Experiments execute Python models validated against native C implementations.
4. There is no compiler-pass instrumentation or sanitizer integration.
5. Reported execution rates represent this experimental implementation rather than production deployment performance.
6. The targets are synthetic and isolate individual sensitivity dimensions rather than modelling real-world CVE distributions.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `no usable Python interpreter found` | Install `numpy`, `scipy`, `matplotlib`, and `pytest`, or specify an interpreter with `AE_PYTHON=/path/to/python ./run_ae.sh --quick`. |
| `gcc: command not found` | Install a C toolchain such as `build-essential` on Debian/Ubuntu or Xcode command-line tools on macOS. |
| `--full` is slow | The full mode re-executes approximately 1,400 fuzzing trials. Increase `--workers` to parallelise trials. |
| Want a clean slate | `make clean` removes generated results. `make distclean` additionally removes compiled `.so` files. |

The workloads are CPU-bound and branch-heavy.

Experiment runners therefore parallelise across processes. No GPU is required.
