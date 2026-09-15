# Installation

The artifact needs a C compiler and four Python packages. Nothing else — no
network access, no data downloads, no external services.

```bash
cd artifact_evaluation
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./run_ae.sh --quick
```

`run_ae.sh` finds `./.venv/bin/python` automatically. Expected wall time for
`--quick` is about 70 seconds.

---

## 1. Requirements

### Hardware

| | `--quick` | `--full` |
|---|---|---|
| CPU | any x86-64 or arm64 core | the more the better; the campaigns parallelise across trials |
| RAM | ~2 GB | ~1 GB per worker process |
| Disk | ~60 MB for the unpacked artifact, plus ~30 MB of generated output | as `--quick`, but `data/` is rewritten |
| GPU | not used | not used |
| Wall time | **~70 s** | hours; scales roughly inversely with `--workers` |

The reference numbers in the paper were produced on a 32-core x86-64 Linux machine.
The 810-trial scheduler matrix took 334 s on 26 workers there.

**No GPU is used.** The workload is branch-heavy interpreted execution driving a
ctypes-loaded C library — there is no dense-array kernel to offload. `--full`
parallelises across processes.

### Software

| Requirement | Version used for the reported results | Minimum |
|---|---|---|
| C compiler | gcc 12.2.0 (Debian 12.2.0-14+deb12u1) | any C99 compiler; clang works |
| Python | 3.12.10 | 3.11 |
| numpy | 2.5.1 | 1.26 |
| scipy | 1.18.0 | 1.11 |
| matplotlib | 3.11.1 | 3.8 |
| pytest | 9.1.1 | 7.4 |

Results are not sensitive to these versions: all randomness comes from explicitly
seeded `random.Random` and `numpy.random.Generator` instances rather than from
library internals.

---

## 2. Installing

Any one of these works. `run_ae.sh` probes candidate interpreters and picks the
first that can import all four packages, so you do not need to tell it which you
chose.

### Option A — venv (recommended)

```bash
cd artifact_evaluation
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### Option B — uv

```bash
cd artifact_evaluation
uv venv
uv pip install -r requirements.txt
```

### Option C — existing environment

```bash
pip install -r requirements.txt
```

### Option D — point at an interpreter you already have

```bash
AE_PYTHON=/path/to/python ./run_ae.sh --quick
```

If `AE_PYTHON` is set but cannot import the dependencies, the runner says so and
exits rather than failing later with a traceback.

### C toolchain

| Platform | Command |
|---|---|
| Debian / Ubuntu | `sudo apt-get install build-essential` |
| Fedora / RHEL | `sudo dnf install gcc` |
| macOS | `xcode-select --install` |

Verify with `gcc --version`. The libraries are built with
`gcc -O2 -fPIC -shared`; `make build` prints the compiler version it used, and it
is also recorded in `results/build_native.json`.

---

## 3. Running

```bash
./run_ae.sh --quick                              # the full evaluation, ~70 s
./run_ae.sh --full --workers 26 --trials 30      # also re-run the campaigns
```

Or drive the stages individually:

```bash
make build       # compile the native C benchmark libraries into native/
make test        # 75 invariant tests
make fidelity    # 20,000 native-vs-Python comparisons
make analysis    # statistics, FDR family, summary tables into results/
make figures     # every figure into figures/ and figures/paper/
make quick       # all of the above via run_ae.sh
make clean       # remove results/ and figures/, keep bundled data/
```

`make` picks `python3` by default; override with `make test PY="./.venv/bin/python"`.

---

## 4. Verifying it worked

`run_ae.sh` prints a per-stage PASS/FAIL table and exits non-zero if any stage
failed. After a successful `--quick` run you should have:

| Path | What to check |
|---|---|
| `results/summary_tables.md` | Table 5 reports 20,000 comparisons / 0 mismatches; Tables 1–2 hold the campaign outcomes |
| `results/ae_summary.json` | `all_documents_valid: true`; `recomputation_check.reproduces_reference: true` |
| `results/fidelity_verification.json` | `fidelity_ok: true`, `n_mismatches_total: 0` |
| `results/run_ae_stages.log` | five `PASS` lines |
| `figures/` | 20 extended figures as PDF + PNG |
| `figures/paper/` | the 5 consolidated submission figures as PDF |

Independent spot checks a reviewer can run directly:

```bash
# the 75 tests, verbosely
python -m pytest tests/test_invariants.py -v

# the fidelity campaign on its own
python tests/test_fidelity.py

# confirm no module opens an absolute path (the package must be relocatable)
grep -rnE '(Path|open)\(\s*"/' --include='*.py' . ; echo "exit $?"
#   -> prints nothing, exit 1 (grep found no matches)
```

`PORT_LOG.json` records the exact edits applied to each research module when it was
moved into this package, so you can confirm the port changed only file locations and
not analysis logic.

---

## 5. If something goes wrong

| Symptom | Fix |
|---|---|
| `no usable Python interpreter found` | Install the dependencies (section 2) or set `AE_PYTHON`. |
| `gcc: command not found` | Install a C toolchain (section 2). |
| `ModuleNotFoundError: No module named 'scipy'` | The interpreter running the stage is not the one you installed into. Use `AE_PYTHON=/path/to/python ./run_ae.sh --quick`. |
| `permission denied: ./run_ae.sh` | `chmod +x run_ae.sh` |
| Stage 4 reports non-finite floats | A result document contains `inf`/`nan`, which RFC 8259 JSON cannot represent. `results/ae_summary.json` names the offending paths under `json_validation`. |
| Want to start over | `make distclean` removes generated output *and* the compiled `.so` files; `data/` is preserved either way. |

The artifact never writes outside its own directory.
