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

### Outcome: the prediction was wrong, in the direction the sibling project warned about

**3 of 25 pathology-cells** reliably degraded held-out performance across both smoke seeds —
12%, against a predicted 30–60%. All three are DQN on CartPole: `P5_no_target_network`,
`P6_stale_target`, `P8_plasticity_loss`. Four more degraded on one seed of two.

That is exactly the pre-registered falsifier's boundary: "fewer than 3 surviving cells and there is
no multi-class attribution problem left". Three is not below the line, but it is on it.

Three separate things went wrong, and only one of them was the hypothesis:

1. **The failure rule was broken and inflated the first count to 11/31.** Comparing with `<=`
   against a floor taken from saturated controls meant a run scoring *identically to the control*
   was labelled a failure. `chain_rho/ppo` controls both scored 1.000, so the floor was 1.000, and
   four pathologies that also scored 1.000 were reported as reliable failure vehicles. Fixed by
   requiring a strict fall below the floor by a margin proportional to the control mean.

2. **The gate never implemented its own first question.** It documented "is the control healthy?"
   and then never checked. `chain_rho/dqn`'s control scored 0.472 against a random policy's 0.483 —
   it had learned nothing at all, and was still being used to derive a floor that five pathologies
   were then measured against. Now every cell is compared to a random-policy baseline and marked
   unusable if it has not beaten it.

3. **The PPO control was under-tuned, so three "pathologies" were upgrades.** On CartPole,
   `P1_entropy_collapse`, `P2_trust_region_blowup` and `P3_obs_norm_freeze` all *beat* the control
   (500.0 against 413/272). A sweep confirmed the baseline was simply bad: `lr=3e-3, ent_coef=0.0`
   reaches 500.0 on 3/3 seeds where the control's `lr=3e-4, ent_coef=0.01` averages 365. If
   injecting a fault improves the score, the baseline was the fault.

The honest reading of (3) is that CartPole is too easy for the entropy bonus to matter, so entropy
collapse is not inducible there by removing it. That is an environment-selection result, not a
detector result, and it belongs to the same family as the sibling project's finding that
collapse-proneness is task-specific.

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

## Phase 2 — outcome: attribution works, early warning does not

Corpus: 63 runs (15 controls, 48 failed pathology runs), 3 seeds, pooled within an algorithm
across environments. Only doses the sweep showed to reliably degrade are included, and a pathology
run that did not degrade is excluded rather than labelled with a mechanism it did not exhibit.

| | pooled DQN | pooled PPO |
|---|---|---|
| leave-one-seed-out accuracy | **0.933** | **0.812** |
| majority class | 0.400 | 0.521 |
| control: step-index only | 0.400 | 0.521 |
| control: train-return only | 0.600 | 0.521 |
| control: shuffled labels | 0.267 | 0.271 |
| margin over step-index | **+0.533** | **+0.292** |
| median lead time @ FPR | **−2,000 steps** | **−3,584 steps** |
| detections that were early | 22% | 0% |

**Predictions that held.** The step-index control landed at 0.400 and 0.521, inside the predicted
0.40–0.60, and the real model cleared it by well over the required 0.15 — the corpus is not merely
time-confounded. Leave-one-pathology-out came in at 0.00–0.39, at or below chance, as predicted: a
detector cannot place a mechanism it has never seen, so the honest claim is "names the failures it
was trained on", not "diagnoses RL runs".

**The falsifier fired.** The pre-registered exit condition was: *median lead time at a fixed
false-alarm rate ≤ 0 across all surviving pathologies*. Measured: −2,000 and −3,584 steps, with
0–22% of detections arriving before the held-out return fell. The detector reliably names the
cause, and it does so **after the reward curve already showed you there was a problem.**

**Why, and this is the interesting part.** The prediction was that lead time would *split* by
failure shape — large and positive for divergence-type faults, near zero for slow degradation.
That was wrong in a way that is more informative than being right. The smoke gate selects for
mechanisms that reliably and quickly destroy a run, and those are precisely the ones with no
warning period to detect: trust-region blow-up at lr=0.1 takes CartPole from 488 to 9.4, which is
the pole falling immediately. The slow mechanisms that *would* have a detectable onset —
update-to-data ratio at 4x, replay staleness, stale targets at 5,000 — never degrade enough to
clear the failure threshold, so they are never labelled failures and never enter the corpus.

So the structural tension is: **the pathologies severe enough to label reliably are too fast to
warn about, and the ones slow enough to warn about are too mild to label.** Every dose in this
sweep sits on one side or the other of that gap. It is not obvious that the gap can be closed by
choosing better doses, because the dose–response curves are cliffs rather than gradients — stale
targets do nothing at 5,000 and destroy the run at 20,000, with nothing in between.

**What is not claimed.** The false-alarm rate is 0.11–0.17 against a 0.05 target. With 15 control
runs you cannot demonstrate a 5% false-alarm rate; the number reported is what the data supports.

## Phase 3 — credit assignment

**Prediction.** On `chain_rho`, GAE's per-step credit will correlate with exact credit at
Spearman ρ > 0.7 at decision density 1.0 and fall below 0.3 at density 0.1, while uniform
trajectory-level credit will be near 0 at both. If GAE is already near-exact at low density, the
diagnostic axis has nothing to measure and Phase 3 is dropped.

**What would falsify it.** Every method scoring approximately the same against exact truth. That
would mean the exact-credit environment does not discriminate, and the contribution evaporates.
