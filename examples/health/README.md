# Health capital & mortality — Grossman (1972), life-cycle form

A life-cycle household that invests in **health capital** (which depreciates) and
faces a **health-dependent mortality hazard**. The household block is built from
**existing library stages only** — no bespoke household stage is rolled here. This
is the Part-3 demonstration that the Grossman demand-for-health model is a `∘` of
catalog stages.

## Household block (existing stages only)

State space: `(:health, :alive)` with `:alive ∈ {1, 0}` (alive, dead).

| Stage (time order) | Library stage | What it does |
|---|---|---|
| `invest` | `CapitalInvestmentStage(:health)` | From health `h` pick next stock `h'`; pay a convex medical-expenditure cost on gross investment `i = h' − (1−δ)h`; earn the health flow `R·h`. Same stage as `examples/human_capital`, different flow (Grossman's value of health). |
| `mortality` | `MarkovStage(:alive)` | Health-dependent sub-stochastic survival. The transition is a dep closure `(; health) -> [survival(h) 1−survival(h); 0 1]` — the dead state is absorbing, and on the alive slice the row sums to `survival(h) < 1`. `survival(h)` is a logistic in health (economic DATA fed to the stage). |

Block: `invest ∘ mortality` — invest in health, **then** survive on the health you
chose. `∘` runs the left stage first (time-ordered). Moments attached via
`define_moments!`: `mean_health`, `survival_rate`, `mean_medical` (all read on the
alive slice).

## Why this is library-stages-only, and finite-horizon

- **Mortality is inside an existing stage.** The catalog ("building a transition
  matrix from economic primitives and handing it to an EXISTING stage is allowed —
  it is data, not a stage") licenses the health-dependent survival matrix fed to
  `MarkovStage`. No bespoke per-cell value/transition logic on the household side.
- **Finite-horizon, not stationary.** A stationary distribution with mortality but
  no birth leaks all mass to the dead state — a nondegenerate stationary alive-mass
  needs a forward mass-injection *source* (catalog gap **G2**, `Λ' = K·Λ + M·g`),
  which is not a household stage. The life-cycle cohort sidesteps G2 honestly: a
  cohort is born alive at health `h0`, invests each age, and its alive-mass traces
  the survival curve. Total mass on the `(health, alive)` grid stays 1 (it
  accumulates in the absorbing dead state).

## Driver (example-side outer loop, allowed)

`steady_state.jl` rolls the finite-horizon solve (this is driver logic, not a
household stage):

1. **Backward induction** — sweep ages `N…1` from a zero terminal continuation,
   seating each age's health-investment policy at its env `(; R, a = efficiency_at_age(t))`.
   The survival `MarkovStage` backward weights the alive continuation by
   `survival(h)` — the Grossman demand-for-health margin (healthier ⇒ more future
   value).
2. **Forward cohort sweep** — a unit point mass born alive at `h0`, pushed age
   `1…N`. Each age, a `1−survival(h)` share of alive mass moves to the absorbing
   dead state.

The age-specific health-production efficiency (declining with age) is threaded
through `env`; the rate `R` (value of health) is exogenous (partial equilibrium —
no market to clear).

## Equilibrium notes / expected output

Baseline (`N_age = 50`, `γ = 0.7`, `δ = 0.08`, `R = 1.0`):

- Total grid mass conserved at `1.0` every age (absorbing dead state).
- Survival declines `1.0 → ≈0.85 (mid-life) → ≈0.003 (end)`; life expectancy
  `≈ 33` ages.
- Mean health among the living is hump-shaped and interior to the grid: birth
  `≈ 6.0`, peak `≈ 20.7` (age 10, grid top 30), end `≈ 2.0`. Health rises early
  (heavy investment when efficiency is high) and declines late (efficiency tapers,
  depreciation dominates) — the Grossman age–health profile that drives rising
  mortality with age.

## How to run

```julia
# from the workspace root (Project.toml develops HouseholdStages)
julia --project=. examples/health/steady_state.jl
```

Regression test: `test/test_example_health.jl` (module-wrapped) — asserts mass
conservation, finite `V`, a decreasing survival curve, sensible life expectancy,
and positive interior mean health.
