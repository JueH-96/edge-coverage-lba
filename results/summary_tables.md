# Artifact-evaluation summary tables

Generated 2026-08-18T00:58:26Z by `experiments/analyze_results.py`.

Each table names the results document it was read from. Documents marked **bundled** below were shipped with the artifact rather than recomputed in this run; see `results/ae_summary.json` for the per-document provenance.

| Result document | provenance |
|---|---|
| `results/corpus_cap_ablation.json` | bundled (from `data/precomputed/corpus_cap_ablation.json`) |
| `results/step1_guidance_validation.json` | bundled (from `data/precomputed/step1_guidance_validation.json`) |
| `results/step2_revision.json` | bundled (from `data/precomputed/step2_revision.json`) |
| `results/step2_scheduler_validation.json` | regenerated (from `data/step2_trials.json`) |

### Recomputation check

`results/step2_scheduler_validation.json` was recomputed here from `data/step2_trials.json` and compared against the copy shipped with the artifact:

- numeric leaves compared: **4168**
- matching: **4168**, differing: **0**
- verdict: **reproduces the shipped analysis**

_Timing fields (wall_seconds and similar) are excluded: they are properties of the machine, not of the analysis._

### Table 5 - Native-C vs Python model fidelity

| Suite | targets | inputs/target | comparisons | mismatches |
|---|---|---|---|---|
| step1_targets | 4 | 2000 | 8000 | 0 |
| trigram_targets | 3 | 4000 | 12000 | 0 |
| **total** | 7 | | **20000** | **0** |

### Table 1 - Step 1 bug-finding success (trials with the bug found / trials)

Blocked design: every configuration sees the same seed sequence, so the only difference between columns is the feedback function.

| Target | Targeted dim | Budget (execs) | blind_random | edge_only | d0_d1_context | d0_d2_valuerange | d0_d3_statemachine | full_3d |
|---|---|---|---|---|---|---|---|---|
| T1_context | D1 | 60000 | 0/30 | 29/30 | 30/30 | 30/30 | 29/30 | 30/30 |
| T2_value_range | D2 | 300000 | 0/30 | 0/30 | 0/30 | 24/30 | 0/30 | 24/30 |
| T3_state_machine | D3 | 60000 | 3/30 | 20/30 | 20/30 | 20/30 | 30/30 | 30/30 |

### Table 2 - Step 2 energy-schedule matrix (bugs found / trials)

Design: 30 trials per cell at 50000 execs, master seed 20260811, seeds blocked across arms. Control = `d3d_rr_n4`, treatment = `d3d_adaptive_n4`.

| Target | `blind_random` | `afl_d0_rr` | `afl_d0_fast` | `d3d_rr_n4` | `d3d_rr_n16` | `d3d_fast_n4` | `d3d_static_n4` | `d3d_adaptive_n4` | `d3d_adaptive_dyn` |
|---|---|---|---|---|---|---|---|---|---|
| TG1_trigram_lock | 0/30 | 0/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 | 1/30 |
| TG2_value_trigram | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| TG3_surprisal_maze | 0/30 | 4/30 | 8/30 | 29/30 | 29/30 | 18/30 | 25/30 | 26/30 | 27/30 |

BH-FDR over the whole family: 192 hypotheses, 64 rejected at q < 0.05.

### Table 6 - Constructive AFL blind-spot witness

Target: `TG1_trigram_lock`. Two inputs are compared:

| Property | Identical across the pair? |
|---|---|
| `bigram_multisets_identical` | yes |
| `symbol_counts_identical` | yes |
| `d0_walk_maps_identical` | yes |
| `trigram_multisets_identical` | no |
| `d1_signatures_identical` | no |

Outcome: witness A reaches milestone 6 (bug = True), witness B reaches milestone 1 (bug = False).

> The pair occupies ONE cell of the edge-coverage partition and TWO cells of the context-sensitive partition, while having opposite bug outcomes. This is the Step 1 refinement theorem made operational: edge coverage cannot in principle supply a gradient here.

### Table 3 - Corpus dilution (median over trials)

`sterile_fraction` is the share of corpus entries that never produced a new coverage element; `energy_gini` measures how unevenly fuzzing energy is spread across the corpus. Rising sensitivity should drive Gini up and sterile fraction down.

| Target | Arm | sterile_fraction (median) | energy_gini (median) | corpus_size (median) |
|---|---|---|---|---|
| TG1_trigram_lock | `blind_random` | 1 | 0 | 1 |
| TG1_trigram_lock | `afl_d0_rr` | 0.4649 | 0.009533 | 224.5 |
| TG1_trigram_lock | `afl_d0_fast` | 0.4686 | 0.08342 | 224 |
| TG1_trigram_lock | `d3d_rr_n4` | 0.4002 | 0.2345 | 8163 |
| TG1_trigram_lock | `d3d_rr_n16` | 0.01064 | 0.7946 | 3.043e+04 |
| TG1_trigram_lock | `d3d_fast_n4` | 0.4009 | 0.2344 | 8164 |
| TG1_trigram_lock | `d3d_static_n4` | 0.3422 | 0.6108 | 7364 |
| TG1_trigram_lock | `d3d_adaptive_n4` | 0.2672 | 0.7347 | 7312 |
| TG1_trigram_lock | `d3d_adaptive_dyn` | 0 | 0.9317 | 2.83e+04 |
| TG2_value_trigram | `blind_random` | 1 | 0 | 1 |
| TG2_value_trigram | `afl_d0_rr` | 0.4623 | 0.01269 | 181.5 |
| TG2_value_trigram | `afl_d0_fast` | 0.4636 | 0.06958 | 180 |
| TG2_value_trigram | `d3d_rr_n4` | 0.4496 | 0.1082 | 7008 |
| TG2_value_trigram | `d3d_rr_n16` | 0.01352 | 0.7741 | 2.767e+04 |
| TG2_value_trigram | `d3d_fast_n4` | 0.4483 | 0.1066 | 6996 |
| TG2_value_trigram | `d3d_static_n4` | 0.3974 | 0.5393 | 6164 |
| TG2_value_trigram | `d3d_adaptive_n4` | 0.3243 | 0.7015 | 6290 |
| TG2_value_trigram | `d3d_adaptive_dyn` | 0 | 0.9314 | 2.816e+04 |
| TG3_surprisal_maze | `blind_random` | 1 | 0 | 1 |
| TG3_surprisal_maze | `afl_d0_rr` | 0.4684 | 0.008274 | 185 |
| TG3_surprisal_maze | `afl_d0_fast` | 0.4611 | 0.06158 | 185.5 |
| TG3_surprisal_maze | `d3d_rr_n4` | 0.5323 | 0.03 | 383 |
| TG3_surprisal_maze | `d3d_rr_n16` | 0.5323 | 0.03 | 383 |
| TG3_surprisal_maze | `d3d_fast_n4` | 0.5349 | 0.1057 | 381.5 |
| TG3_surprisal_maze | `d3d_static_n4` | 0.7052 | 0.1804 | 395.5 |
| TG3_surprisal_maze | `d3d_adaptive_n4` | 0.7247 | 0.3657 | 370 |
| TG3_surprisal_maze | `d3d_adaptive_dyn` | 0.7622 | 0.4344 | 602.5 |

### Table 4 - T4 n-gram order sweep (round-robin energy)

Guidance of order k should discriminate the target exactly when k is at least the order the bug is built on; edge coverage (`edge_only`) is a bigram statistic and is predicted to be blind.

Target `T4_trigram`, schedule `round_robin`, 30 trials/cell at 400000 execs.

| Config | n-gram order | successes | success rate | median execs to bug |
|---|---|---|---|---|
| `blind_random` | 0 | 0/30 | 0.000 | n/a (no successes) |
| `edge_only` | 2 | 15/30 | 0.500 | 213055.0 |
| `ngram2` | 2 | 15/30 | 0.500 | 213055.0 |
| `ngram3` | 3 | 26/30 | 0.867 | 177619.5 |
| `ngram4` | 4 | 8/30 | 0.267 | 230549.0 |
| `ngram6` | 6 | 5/30 | 0.167 | 188411.0 |
| `ngram8` | 8 | 2/30 | 0.067 | 150385.5 |
| `ctx2` | 2 | 10/30 | 0.333 | 168582.0 |
| `ctx3` | 3 | 16/30 | 0.533 | 234100.5 |
| `ctx4` | 4 | 3/30 | 0.100 | 93358.0 |
| `ctx6` | 6 | 2/30 | 0.067 | 136654.0 |
| `ctx8` | 8 | 4/30 | 0.133 | 166979.5 |
| `ctx16` | 16 | 4/30 | 0.133 | 166979.5 |
| `full_3d` | 16 | 4/30 | 0.133 | 166979.5 |
