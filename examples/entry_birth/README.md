# Entry / birth & population mass (`EntryStage`)

The minimal demonstrator that **birth ≠ death moves the population**. A
bare consumption–savings core is wrapped in the demographics composite —
deaths via `ExogenousExit`, newborns via `EntryStage` — and the driver
shows how the stationary population mass depends only on the
birth/death balance.

## The chain

```
IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ ExogenousExit ∘ Entry
```

- **`ExogenousExit`** — survival `s`; forward `Λ ↦ s·Λ` (mass leaks at
  rate `1−s`). Needs `:exiting => Discrete([0])` in the layout (size 1).
- **`EntryStage`** — additive newborn source `Λ ↦ Λ + g`, with newborns
  at zero wealth and the ergodic income draw, total mass `Σg = birth_mass`.

```julia
hh = shock ∘ receipt ∘ savings ∘ exit ∘ entry
```

## The mass dynamics

Forward, the chain's effect on the *total* population collapses to the
affine recursion

```
M_{t+1} = s·M_t + Σg ,      fixed point   M* = Σg / (1 − s).
```

So the stationary population is a pure function of the birth/death
balance, independent of anything in the savings problem:

| regime | `Σg` | stationary mass `M*` |
|---|---|---|
| replacement | `1 − s` | `1` |
| birth-heavy | `2(1 − s)` | `2` |

Household policies do not depend on the aggregate mass, so the savings
policy (hence `V`) is solved once and the mass dynamics are a linear
afterthought — which is the whole point of the demonstrator.

## Affine, not geometric

Because `EntryStage` is a **fixed** additive source (the inflow `Σg` does
not scale with the current population), the mass map is affine, not
geometric. A birth-heavy regime converges to a higher stationary *level*,
not an unbounded growth path; the per-pass mass ratio `M_t/M_{t-1}`
relaxes to 1 as the level approaches `M*`. The driver prints this
trajectory starting from the replacement steady state (`M₀ = 1`):

```
pass  1:  M = 1.050000   (M_t/M_{t-1} = 1.05000)
pass  2:  M = 1.097500   (M_t/M_{t-1} = 1.04524)
...
pass 12:  M = 1.459640   (M_t/M_{t-1} = 1.01987)   → climbing toward M* = 2
```

Genuine exponential population growth (a constant growth factor `γ > 1`)
would require a source proportional to the current mass, `Σg = (γ−s)·M_t`
— which a fixed additive `g` deliberately does not provide. That is a
modelling choice of `EntryStage`, not a limitation worked around here.

## Result

Default `EntryBirthParams`: β = 0.95, σ = 1.0 (log), s = 0.95; 2-state
income on `[0.7, 1.3]`; `N_w = 120` on `[0, 60]`. Partial equilibrium at
`r = 0.03`, `w = 1.0`.

```
Regime (i)  replacement   Σg = 1−s     ⇒ ΣΛ = 1.000000
Regime (ii) birth-heavy   Σg = 2(1−s)  ⇒ ΣΛ = 1.999955   (analytic 2.0)
mean wealth E[w] = 0.1151 in both regimes (identical per-capita policy)
```

Regime (i) and (ii) are each a single `solve_steady_state_given_env!`
(the affine map with `s < 1` is a contraction, so both have a finite
stationary distribution). The transient trajectory uses the block's own
`forward!` directly from the driver.

## Run

```bash
julia --project=. examples/entry_birth/steady_state.jl
```

About 11 seconds including first-call compilation.

## Files

- `model.jl` — parameters, ergodic-income helper, layout, the five-stage
  chain with a tunable `birth_mass`.
- `steady_state.jl` — the two regimes and the regime-(ii) mass trajectory.
