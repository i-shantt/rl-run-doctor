# Where this is

Updated 2026-08-01.

## Built and passing

| Piece | Status |
|---|---|
| `testbed/envs/` | CartPole, DeepSea, ChainRho. Determinism and the exact-credit property are tested, not assumed. |
| `testbed/algos/` | PPO and DQN, instrumented for the signature battery. Controls tuned (see below). |
| `testbed/inject/` | 11 pathology families as dose ladders; every level checked to build against its target config. |
| `testbed/telemetry.py` | `signals` / `oracle` split, enforced by the loader rather than by discipline. |
| `testbed/health.py` | Windowed health, control-derived failure band, random-policy reference. |
| `src/rl_run_doctor/` | Causal shape features, numpy-only linear diagnoser, lead-time-at-fixed-FPR, credit estimators. |
| `scripts/` | Smoke gate (with `--report-only`), corpus builder, detector fitting, credit study. |
| CI | ruff + pytest + an assertion that the shipped package imports without torch. |

43 tests. No numbers in the README, by design, until the experiments that produce them have run.

## Measured so far

**Controls.** PPO on CartPole at `lr=1e-3` averages 464 over 3 seeds (worst 392). DQN on CartPole
averages 322 over 6 seeds with a range of 118–500 — that band is wide and it is DQN, not a bug;
training it longer makes the mean *worse* (263 at 100k). Scoring a single final evaluation instead
of a trailing window inflated the across-seed standard deviation from 68 to 161 on the same runs,
which is why health is a window.

**First smoke gate (2 seeds, single dose per pathology).** 3 of 25 cells reliably degraded, against
a pre-registered 30–60%. All three were DQN on CartPole. See `NEGATIVE_RESULTS.md` for the
prediction and the outcome side by side.

Two bugs inflated the first reading to 11/31, and both made things look better rather than worse:

* the failure rule compared with `<=` against a floor taken from saturated controls, so a run
  scoring *identically to the control* counted as a failure;
* the gate documented a control-health check and never performed one, so `chain_rho/dqn` supplied a
  floor from a control that had not beaten a random policy (0.472 against 0.483).

A third problem was not a bug in the gate but in the baseline: on CartPole three PPO "pathologies"
scored better than the control, because the control was under-tuned.

## Running now

Severity sweep: 246 runs, 3 seeds, dose ladders per family, 4 workers. Replaces the binary
keep/drop verdict with a dose-response curve per mechanism.

## Next, in order

1. Read the dose-response curves; decide which (cell, family, dose) combinations enter the corpus.
2. Build the corpus from surviving cells, controls at double the seeds.
3. Fit the attribution model. The numbers that decide whether it is real are the step-index-only
   control and leave-one-pathology-out, not the accuracy.
4. Credit study on `chain_rho` against exact per-step truth.

## Conventions

* Free compute only; CPU testbed. Ask before using any Kaggle quota.
* Shipped package depends on numpy alone. `torch` is a testbed extra.
* Traces are flushed as produced, so analysis is always replayable from disk — a reporting bug
  costs seconds, not another sweep. This has already paid for itself once.
* Predictions go in `NEGATIVE_RESULTS.md` before the run that tests them.
