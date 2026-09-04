# Gaussian / variance rational inattention — savings + attention

An incomplete-markets savings household that **also chooses the dispersion of its next-period wealth
state at an information cost** — the Sims (2003) / Maćkowiak–Wiederholt (2009) variance-RI reading.
The within-period problem is **four existing `HouseholdStages` stages, no bespoke household stage** —
the demonstration this example exists for (Part 3: literature household blocks expressible from the
library alone).

## Household block

In time order:

```
IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Attention
```

| stage | library stage | does |
|---|---|---|
| `IncomeShock` | `MarkovStage` | draw next income on the `:income` axis |
| `Receipt` | `WealthChangeStage` | `b ↦ (1+r) b + w·y` |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | pick next wealth `b'`, `c = b_in − b'`, CRRA utility |
| `Attention` | `MeanPreservingSpreadStage` | pick the continuous dispersion `θ ∈ [0, θ_max]` of a Gaussian mean-preserving spread of `b'` (sd `θ`, clamped) at cost `c(θ) = λ·θ²` |

`MeanPreservingSpreadStage` is the continuous variance/MPS primitive (`O(n_w)`, no θ-axis): per cell
it solves the smooth Gaussian-spread objective for `θ*` (internal scan + Newton), seats the per-cell
optimal `θ*(x)`, and pushes mass through the chosen
mean-preserving spread. `θ = 0` is **perfect attention** (no noise, no cost); larger `θ` is a noisier
signal about the state. The cost `c(θ) = λ·θ²` is a **stand-in for `λ·KL(θ)`** — a quadratic
placeholder for the information cost, not a literal Shannon mutual-information term.

## Why θ\* is interior (the economics)

A pure mean-preserving spread on a globally concave continuation always picks `θ = 0` (Jensen, plus a
positive cost). The bite here is the **borrowing constraint**: for poor, near-constrained households
the continuation value in wealth is locally **convex** (a low draw is floored at the constraint, a
high draw escapes it), so a small dispersion carries option value. The quadratic information cost
`λ·θ²` then disciplines how much is taken. The result is a sensible attention gradient — `θ*` high for
the constrained poor, `→ 0` for the wealthy — and `θ*` falls everywhere as `λ` rises. This is the
classic option-value-of-risk-near-a-constraint mechanism standing in for the RI signal-precision
choice; it gives a non-degenerate, interpretable policy from library stages alone.

## Equilibrium

Returns are **exogenous** (partial equilibrium), so there is no market to clear: the outer loop is a
single `solve_steady_state_given_env!`. The borrowing constraint (`b' ≥ 0`, the grid floor) plus
impatience (`β·(1+r) < 1`) give a stationary wealth distribution.

## Run

```julia
julia --project=. examples/rational_inattention/steady_state.jl
```

Reports mass conservation, mean wealth, the attention (dispersion) policy range, and the **RI
comparative static**: mean chosen `θ*` falls monotonically as the information cost `λ` rises.
