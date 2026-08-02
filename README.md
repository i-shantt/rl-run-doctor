# rl-run-doctor

**Names why a reinforcement-learning run failed, from its logs and checkpoints.**

An RL run dies and the reward curve just goes flat. Was it value overestimation? Entropy collapse?
A dead network? Stale replay? A seeding bug that made your five "independent" seeds one seed?
The failure modes are distinct, individually measurable, and well characterised in the literature —
and no installable tool looks at a run and tells you which one you have.

```python
from rl_run_doctor import diagnose

report = diagnose("runs/ppo_cartpole_seed3/")
print(report.verdict)
# → value_overestimation (confidence 0.81), onset ≈ step 41_200
#   evidence: TD-error mean drift +3.2σ, value-target gap widening, |Q| growth 4.1×
```

> **Status: pre-alpha, nothing measured yet.** No numbers appear in this README because the
> experiments that would produce them have not been run. Everything reported here will be generated
> from committed run artifacts, and CI will assert the generated file matches.

---

## Why this exists

`google-research/rliable` — the reference implementation from a NeurIPS 2021 Outstanding Paper — was
**archived in August 2024** and still serves roughly 9,000 downloads a month. A GitHub search for
`reinforcement learning debugging diagnostics` returns, as its top result by stars, a repository
with **one star**. Deep learning has profilers and weight analysers; RL, whose failure modes are
*more* distinctive, has nothing.

The metrics themselves are not the contribution — dormant-neuron fraction, effective rank, churn and
the rest are published work, and are lifted rather than reinvented. The contribution is:

1. a corpus of runs where the failure cause is known **by injection**, not by guesswork;
2. an attribution layer mapping signature trajectories to named causes;
3. evaluation on **lead time at a fixed false-alarm rate** — how much warning you get before the
   return actually degrades — which no existing RL diagnostic reports.

## The thing most likely to kill this project

The premise is that injecting a pathology produces a labeled failure. A sibling project measured
that this is *mostly false* in the RLVR setting: across 26 induced conditions, only two collapsed
reliably, and collapse-proneness turned out to be strongly task-specific. There is no reason to
assume PPO/DQN/SAC are friendlier.

So the corpus is gated. Before any large run, a per-cell smoke test checks that each
`(env, algorithm, pathology)` cell both **keeps its control healthy** and **reliably degrades under
injection**. Cells that fail are dropped and reported, not quietly retried. If almost nothing
collapses, that is itself the finding and this becomes a paper about why the field's failure
taxonomy is not identifiable from telemetry.

## Design constraints

- **`numpy` only** in the shipped package (`src/rl_run_doctor/`). It is meant to install *into* an
  existing training image without dragging in a tensor library. `torch` is a testbed extra, and CI
  asserts it never enters `sys.modules` on import.
- **Free compute only.** The testbed runs on CPU. Nothing here requires a GPU.
- Ground truth is always a **held-out evaluator the training signal cannot touch** — never the
  training return.

## Layout

| Path | What |
|---|---|
| `src/rl_run_doctor/` | the shipped package: signatures, attribution, evaluation |
| `testbed/` | CPU RL testbed that generates the labeled corpus (torch) |
| `scripts/` | corpus build, smoke gate, reporting |
| `docs/` | design notes, negative-result predictions |

## License

MIT
