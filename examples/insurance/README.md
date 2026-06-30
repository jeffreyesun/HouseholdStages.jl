# Insurance / annuitization — convex-cost loss insurance

An incomplete-markets savings household that can pay a **convex premium** to insure against a loss
shock on its asset stock. Each period, before income is received, the household chooses how much
coverage `θ ∈ [0,1]` to buy against a casualty/health-type loss that would shrink its wealth by a
`loss_factor`. The within-period problem is **four existing `HouseholdStages` stages, no bespoke
household stage** — the demonstration this example exists for (Part 3: literature household blocks
expressible from the library alone). It is the Yaari (1965) annuity / standard insurance-demand
block, and the "pay a convex cost to mix toward a safe kernel" reading of Grossman-type mortality
retention.

## Household block

In time order:

```
IncomeShock ∘ Insurance ∘ Receipt ∘ ConsumptionSavings
```

| stage | library stage | does |
|---|---|---|
| `IncomeShock` | `MarkovStage` | draw next income on the `:income` axis |
| `Insurance` | `MixingStage` (via `RetentionStage`) | blend a no-loss kernel `K_A = I` and a loss kernel `K_B` (asset stock `× loss_factor`) at convex cost `c(θ) = θ²/(2κ)`; seat coverage `θ*(x)` |
| `Receipt` | `WealthChangeStage` | `a ↦ (1+r)·a + w·y` (cash-on-hand, on the post-loss stock) |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | pick next wealth `b'`, `c = x − b'`, CRRA utility |

`Insurance` is a `RetentionStage` — the `MixingStage` special case with `K_A = I` ("pay not to
transition"). Its closed form is `V = K_B·V + c*(K_A·V − K_B·V)`, the Fenchel conjugate of the
quadratic premium (two Markov applies + a pointwise conjugate, no θ-axis grid). The loss kernel
`K_B` is plain economic data — the row-stochastic, on-grid, mass-conserving discretization of the
deterministic map `x ↦ loss_factor·x`, built in `model.jl`'s `loss_kernel` (a driver helper, not a
stage).

**Timing / why insurance sits before receipt.** The loss hits *beginning-of-period* asset wealth,
which is the canonical casualty/health-loss timing. It also keeps the `MixingStage` closed form
well-defined: `ConsumptionSavingsStage` emits `-Inf` at the wealth-grid floor (no feasible
consumption when `b' = b_in` at `b_in = w_min`), and `K_A·V − K_B·V` on `-Inf` entries would be
`NaN`. Placing `Receipt` between savings and insurance means the post-receipt gather maps those
floor cells to interior points, so `Insurance` always sees a finite continuation value.

## Equilibrium

Returns and the wage are **exogenous** (partial equilibrium), so there is no market to clear: the
outer loop is a single `solve_steady_state_given_env!`. The grid floor (`b' ≥ 0`) plus impatience
(`β·(1+r) < 1`) give a stationary wealth distribution.

## Run

```julia
julia --project=. examples/insurance/steady_state.jl
```

Reports mass conservation, mean wealth, and the coverage policy range. With the default calibration
(`σ = 2`, 30% loss, `κ = 4`) coverage is interior and tracks the standard insurance-demand
comparative statics: a **larger loss** raises coverage (30%→50% loss: mass-weighted `θ` 0.64→0.72),
a **more expensive premium** collapses it (`κ` 4→1: `θ` 0.64→0.01), and a **smaller loss** lowers it
(10% loss: `θ` 0.23). Wealthy cells almost always fully insure (`θ → 1`); the borrowing-constrained
buy less, having less stock at risk.
