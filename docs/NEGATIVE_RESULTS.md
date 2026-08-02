# Predictions, written before the experiments

Recorded so that a disappointing result cannot be quietly reframed as an expected one. Each entry
states what would count as failure, in numbers, before the run that decides it.

Written 2026-08-01. The smoke gate was executing when this was written and its output had not been
read; everything below Phase 0 has not been run at all.

---

## Phase 0 — smoke gate

**Prediction.** Between 30% and 60% of pathology-cells will reliably degrade held-out performance
across both seeds.

The reasoning: a sibling project on RLVR found only 2 of 26 induced conditions collapsed reliably,
but its pathologies were reward-signal perturbations, which a robust optimiser absorbs. Several of
the pathologies here are *mechanically* destructive — removing the target network deletes the
stabiliser that makes bootstrapping convergent — so classic RL should be more fragile than RLVR,
not less.

**What would falsify the project premise.** Fewer than 3 surviving cells in total. With fewer than
3 there is no multi-class attribution problem left, only a binary "something is wrong" detector,
and the interesting claim is gone. In that case the honest output is a short paper reporting that
the field's failure taxonomy is not reliably inducible, not a library.

**Already observed before writing this** (from validation runs, not the gate itself): `chain_rho`
under `P11_credit_dilution` scored 1.000, identical to control. Diluting decision density to a
single decision step did not dent it, because the cue is in the observation and the task stays
trivially learnable. `chain_rho` is therefore expected to be a dead cell for pathology purposes,
and is retained only for its Phase-3 role.

## Phase 1 — corpus

**Prediction.** Healthy control bands will be wide for DQN (measured: 6-seed windowed range
118–500 on CartPole) and narrow for PPO. Consequently, roughly twice as many pathologies will
clear the failure threshold under PPO as under DQN, purely because DQN's own variance hides them.

**What would falsify it.** If the DQN control band is so wide that no pathology clears it, DQN
cannot host a labelled corpus at this scale and the project narrows to PPO — which also removes
the target-network and replay pathologies, i.e. the most distinctive classic-RL failures. That is
a serious but survivable outcome, and it must be stated rather than hidden by quietly dropping DQN.

## Phase 2 — attribution

**Prediction: the step-index-only control will be embarrassingly strong.** A classifier given
nothing but "how far into training are we" should reach 0.4–0.6 accuracy on a balanced multi-class
problem, because injected pathologies bite at characteristic times. If the real detector does not
beat it by at least 0.15 absolute accuracy, the corpus is time-confounded and the headline number
is an artifact.

**Prediction: leave-one-failure-mode-out will be near chance.** Held-out-mode accuracy will land
within 0.1 of chance. Signatures are expected to be mode-specific — dormant units for plasticity
loss, Q-growth for overestimation — so a detector should have no basis for classifying a mechanism
it has never seen. If this holds, the honest claim is "names the failures it was trained on", not
"diagnoses RL runs", and the README must say so.

**Prediction: lead time will split by failure shape.** Divergence-type failures (no target network,
trust-region blow-up) will give large positive lead time — tens of percent of the run — because the
signature moves long before the return does. Slow-degradation failures (replay staleness,
plasticity loss) will give lead time near zero or negative: by the time the signature is
distinguishable, the return has already fallen.

**What would falsify the tool's usefulness.** Median lead time at 5% false-alarm rate ≤ 0 across
all surviving pathologies. A detector that fires only after the reward curve has already dropped
tells the practitioner nothing they did not have, and the project should be reported as a negative
result rather than shipped.

## Phase 3 — credit assignment

**Prediction.** On `chain_rho`, GAE's per-step credit will correlate with exact credit at
Spearman ρ > 0.7 at decision density 1.0 and fall below 0.3 at density 0.1, while uniform
trajectory-level credit will be near 0 at both. If GAE is already near-exact at low density, the
diagnostic axis has nothing to measure and Phase 3 is dropped.

**What would falsify it.** Every method scoring approximately the same against exact truth. That
would mean the exact-credit environment does not discriminate, and the contribution evaporates.
