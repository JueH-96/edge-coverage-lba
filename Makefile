# Artifact evaluation for "Multidimensional Guidance for Coverage-Guided Fuzzing".
#
#   make quick     everything a reviewer needs, from bundled data      (< 2 min)
#   make full      additionally re-run the fuzzing campaigns           (hours)
#
# See README.md for the claim-to-artifact mapping and INSTALL.md for setup.

# Interpreter: uses uv when a lockfile is present, otherwise plain python3.
# Override with e.g.  make quick PY="python3.12"
ifneq ("$(wildcard uv.lock)","")
  PY ?= uv run python
else
  PY ?= python3
endif

WORKERS ?= 12
TRIALS  ?= 30
EXECS   ?= 50000
BUDGET  ?= 150000

.PHONY: help build test fidelity quick full figures analysis clean distclean

help:
	@echo "targets:"
	@echo "  make build     compile the native C benchmark libraries into native/"
	@echo "  make test      run the 75-test invariant suite"
	@echo "  make fidelity  run the 20,000-comparison native-vs-Python check"
	@echo "  make analysis  regenerate statistics and summary tables into results/"
	@echo "  make figures   regenerate every publication figure into figures/"
	@echo "  make quick     the whole evaluation from bundled data (< 2 minutes)"
	@echo "  make full      as above, but re-run the fuzzing campaigns (hours)"
	@echo "  make clean     remove generated results and figures"
	@echo "  make distclean also remove the compiled .so files"
	@echo ""
	@echo "variables: PY=$(PY) WORKERS=$(WORKERS) TRIALS=$(TRIALS) EXECS=$(EXECS)"

# --- individual stages ------------------------------------------------------ #
build:
	$(PY) experiments/build_native.py

test: build
	$(PY) -m pytest tests/test_invariants.py -q

fidelity: build
	$(PY) tests/test_fidelity.py

analysis:
	$(PY) experiments/analyze_results.py

figures: analysis
	$(PY) plots/generate_all_figures.py

# --- one-click entry points ------------------------------------------------- #
quick:
	./run_ae.sh --quick

full:
	./run_ae.sh --full --workers $(WORKERS) --trials $(TRIALS) \
		--execs $(EXECS) --budget $(BUDGET)

# --- housekeeping ----------------------------------------------------------- #
# `clean` deliberately leaves data/ alone: it holds the bundled trial records the
# quick evaluation depends on, and they are not regenerable in under two minutes.
clean:
	rm -rf results/*.json results/*.md results/*.log
	find figures -name '*.pdf' -delete -o -name '*.png' -delete
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +
	@echo "cleaned results/ and figures/ (bundled data/ left intact)"

distclean: clean
	rm -f native/*.so
	@echo "removed compiled shared libraries"
