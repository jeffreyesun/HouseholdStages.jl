# Menzio–Shi (2011) — Directed Search On and Off the Job

**Both** the unemployed and the employed direct their search across
posted-wage submarkets, trading a higher target wage against a lower fill
probability; an employed searcher who fails to match keeps their current job.
From "Efficient Search on the Job and the Business Cycle" (JPE 2011).

The distinguishing feature versus `../directed_search` (where the employed
are **locked**) is **on-the-job search with a fall-back to the current wage**.
The household block is still **existing library stages only** — no bespoke
stage — using two discrete axes to make the on-the-job match expressible.

## Household block

Within-period decomposition, in time order:

```
Aim ∘ Match ∘ Receipt ∘ ConsumptionSavings
```

State axes: `:submarket` (current job — `unemployed` or an employed wage
tier) and `:target` (the wage tier aimed at this period).

| Stage | Library stage | What it does |
|---|---|---|
| `Aim` | `LogitChoiceStage` (axis `:target`) | Choose which posted-wage submarket to apply to. The fill/wage tradeoff lives in the target's continuation value, not a kwarg; the cost is flat (cancels in the softmax). |
| `Match` | `MarkovStage` (axis `:submarket`, transition varies along `:target`) | Given target wage `w_t` with fill prob `f(w_t)`: unemployed match → employed at `w_t` w.p. `f`; employed separate w.p. `s`, else upgrade to `w_t` w.p. `f` **only if it beats the current wage** (on-the-job search), else keep the job. |
| `Receipt` | `WealthChangeStage` (axis `:wealth`) | `(1+r)·a + income` (current wage, or benefit `b` if unemployed). |
| `ConsumptionSavings` | `ConsumptionSavingsStage` (axis `:wealth`) | Saving/consumption on the log-spaced wealth grid. |

The choice-conditional probabilistic move — match to the *aimed* submarket
with prob `f`, fall back to the *current* job otherwise — is the Menzio–Shi
on-the-job search, expressed as a `MarkovStage` transition reading the
`:target` dep. No bespoke kernel.

## Equilibrium notes

Partial equilibrium: prices `(r, b)` and the fill schedule `f(·)` (the
free-entry tightness summary) are exogenous. Single
`solve_steady_state_given_env!`.

Headline result (6 posted-wage tiers `[0.8 … 1.3]`, decreasing fill schedule,
`sep = 0.04`, `ε = 0.05`):

```
unemp rate     = 0.050
mean wage(emp) = 1.25
mass at tier   : unemp 0.05 | 0.80→0.01  0.90→0.04 … 1.20→0.06  1.30→0.77
```

The wage distribution piles up at the **top** tier: workers climb the job
ladder through repeated on-the-job upgrades — the canonical Menzio–Shi result.

## How to run

```bash
julia --project=. examples/menzio_shi/steady_state.jl
```
