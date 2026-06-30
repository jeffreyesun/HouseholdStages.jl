# Skill depreciation during unemployment (Ljungqvist–Sargent 1998 / Pissarides 1992)

## Mechanism

Human capital (skill) **decays while unemployed** and **grows/persists while
employed**. Because the employed wage scales with skill (`income = w · skill`), a
spell of unemployment erodes the worker's re-employment wage — which is exactly what
sharpens search incentives in Ljungqvist & Sargent (1998 JPE) and Pissarides (1992
QJE). Layered on top is a standard income-fluctuation savings spine.

State `= (:wealth [continuous log grid], :skill [5-rung ladder], :emp ∈ {:unemp, :emp})`.

## The exact `∘` chain

The household block is a pure composition of **existing** library stages (`∘` runs
the LEFT stage first):

```
Matching ∘ SkillShock ∘ Receipt ∘ ConsumptionSavings
```

| Stage | Library stage | Role |
|---|---|---|
| `Matching`            | `SearchMatchingStage` (axis `:emp`) | Unemployed choose search effort `e` at cost `χe²/2`, find a job w.p. `1−exp(−A·e·θ)`; employed separate at `δ`. Tightness `θ` fixed (partial equilibrium). `:wealth`/`:skill` ride as spectators. |
| `SkillShock`          | `MarkovStage` (axis `:skill`)       | EMPLOYMENT-DEPENDENT dep closure `(; emp) -> emp == :unemp ? T_decay : T_grow`. `T_decay` drifts skill down one rung; `T_grow` drifts it up. |
| `Receipt`             | `WealthChangeStage` (axis `:wealth`)| `b ↦ (1+r)b + (emp==:emp ? w·skill : b_u)`. Employed income scales with skill. |
| `ConsumptionSavings`  | `ConsumptionSavingsStage` (axis `:wealth`) | Pick `b'`; `c = b_in − b'`; CRRA. |

`T_decay`/`T_grow` are banded row-stochastic matrices built by the plain helper
`skill_drift_kernel` — economic **data** fed to `MarkovStage`, not a new stage. This is
the same `(; dep) -> T` dep-closure pattern as `examples/health` (keyed there on
`:health`, here on `:emp`).

## Fidelity note

This is the Ljungqvist–Sargent skill-loss channel embedded in a Krusell–Mukoyama–Şahin
(2010 ReStud) search-and-savings household. Simplifications relative to the literature:
tightness is a fixed scalar (no free-entry GE — contrast `examples/search_matching`,
which closes `θ`); the skill ladder is a coarse 5-rung Markov drift rather than a
continuous learning-by-doing/atrophy process; and the wage is linear in skill. The
qualitative mechanism — skill erosion in unemployment, hence a lower re-employment
wage and stronger search incentives — is faithful and shows up in the moments.

## Status

**Solves cleanly** (`julia --project=. examples/skill_depreciation/steady_state.jl`).
Representative stationary steady state (default params, θ = 1):

```
mass(Λ)                 = 1.000000
employment rate         = 0.8327
mean wealth             = 3.4064
mean skill | employed   = 1.3119
mean skill | unemployed = 1.1687   (LOWER ⇒ decay mechanism)
mean unemployed effort  = 1.1493
```

Mean skill among the unemployed sits below that among the employed — the depreciation
mechanism, read straight off the stationary distribution.
