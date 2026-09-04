# Risk-shifting / gambling-for-resurrection — Vereshchagina–Hopenhayn (2009)

A borrowing-constrained, limited-liability **entrepreneur who gambles for
resurrection**: where the value function is convex — near the limited-liability
floor a poorly-capitalized entrepreneur faces — the agent picks a *riskier*
project, even at no higher mean return. **Risk-taking is decreasing in net
worth** (Vereshchagina & Hopenhayn, "Risk Taking by Entrepreneurs", *AER* 2009).

The point of this example: the entire within-period problem is **five existing
library stages**, in time order, with **no bespoke household stage**.

## Household block (existing stages only)

| Stage (this example) | Library stage | Role |
|---|---|---|
| `OccShock` | `MarkovStage` (axis `:z`) | Persistent entrepreneurial-productivity shock. |
| `Receipt` | `WealthChangeStage` `a ↦ a + w` | Per-period cash injection — spreads the stationary distribution. |
| `Savings` | `ConsumptionSavingsStage` | Choose the stake `a'` carried into the project; `c = x − a'`, CRRA. Grid floor `a' ≥ a_min` is the borrowing constraint. |
| `Gamble` | `GaussianLoadingStage` (axis `:wealth`; the portfolio stage read as a project-risk dial: anchor = `R_f`, increment = the project's excess payoff) | Pick project risk `θ ∈ [0, 1]` continuously: stake → `a'·(R_f + θ·(μ_x + σ_x·Z))`, the Gaussian excess moment-matched to the succeed/fail bet. **Mean-neutral** (`μ_x ≈ 0`) so convexity, not a premium, drives gambling. |
| `LimitedLiability` | `WealthChangeStage` `a ↦ max(z·a, a_floor)` | Limited liability / outside option: a failed gamble cannot push wealth below `a_floor`. This floor **convexifies V** just above `a_floor` — the engine of risk-shifting. |

Composed as `OccShock ∘ Receipt ∘ Savings ∘ Gamble ∘ LimitedLiability`.

### Why mean-variance suffices (no skewness primitive)

The V–H risk-shifting result is driven by **value-function convexity** induced
by the limited-liability floor, interacting with a **mean-variance** project-risk
choice. The canonical V–H gamble is a *two-point* project (succeed big / fail to
the floor); `GaussianLoadingStage` carries that project through its first two
moments (`μ_x = 0.02`, `σ_x = 0.59`) via its Gaussian return contract. That is
enough: risk-shifting does not live in the bet's *shape* — it is
a **variance dial pulled at a convex region of V**, and the moments are what the
dial controls. The right-tail emphasis is not a primitive skewness control; it
emerges *endogenously* from the floored continuation. So the model needs no
bespoke stage and no streaming skewness primitive — `GaussianLoadingStage` + a
`WealthChangeStage` floor is the faithful statement.

### Why no occupation axis

The V–H floor is a floor on **realized wealth** — limited liability, not a
choice between occupations. A `WealthChangeStage` (`max(z·a, a_floor)`) states
that directly and library-only: the faithful statement of the same convexity
(V–H §II reads the outside option as a lower bound on the continuation; here it
is a lower bound on the carried collateral), with no occupation axis to carry.

## Equilibrium

Returns, productivity, and the wage are **exogenous** (partial equilibrium):
there is no market to clear, so the "outer loop" is a single
`solve_steady_state_given_env!`. The limited-liability floor plus the cash
injection deliver a stationary wealth distribution. That distribution sits low
and churns near `a_floor`: the Gaussian project has an unbounded left tail, and
since the poor gamble at the cap (`θ* = 1`), a bad draw wipes the stake down to
the floor rather than to some bounded fraction of it. Wealth accumulates only
through the cash injection `w` and the runs of good draws that outpace it, so
mean wealth stays close to the floor.

## Parameters (defaults)

`β = 0.96`, `σ = 2` (CRRA), `w = 0.40` cash injection; 2-state productivity
`z ∈ {0.98, 1.02}`; project `R_f = 1.06` (safe), succeed/fail calibration
`(R_up, R_dn) = (1.80, 0.60)` with `p_up = 0.40` ⇒ `E[R_k] = 1.08 ≈ R_f`
(mean-neutral), moment-matched to the Gaussian excess `(μ_x, σ_x) =
(0.02, 0.59)`; continuous risk `θ ∈ [0, 1]`; floor `a_floor = 0.50`; wealth
grid `N_a = 200`, log-spaced on `[0, 200]`.

## Expected output

```
Risk-shifting steady state (σ = 2.0, a_floor = 0.50, E[R_k] = 1.080 vs R_f = 1.06)
  mass(Λ)               = 1.000000
  mean wealth           = 0.9822
  project risk θ*: poor = 1.000  (near floor a≈0.50)
  project risk θ*: rich = 0.019  (top decile a≈200)
  ⇒ risk-shifting: poor gamble 51.9× more than rich
```

The seated policy `θ*(x)` is at the cap near the floor (gambling for
resurrection) and near zero for the well-capitalized — the V–H comparative
static. The rich are not exactly at zero: the calibration is only *nearly*
mean-neutral (`μ_x = 0.02 > 0`), and a continuous `θ` resolves the resulting
Merton demand `≈ μ_x/(σ_CRRA·σ_x²) ≈ 0.03` rather than rounding it off.

## How to run

```julia
julia --project=. examples/risk_shifting/steady_state.jl
```

Or interactively:

```julia
include("examples/risk_shifting/steady_state.jl")
risk_shifting_steady_state()           # full report + (V, Λ, moment, policy)
```

Regression test: `test/test_example_risk_shifting.jl` (module-wrapped) asserts
`mass ≈ 1`, finite `V`, `mean_wealth > a_floor`, valid shares, and the V–H
signature `θ*(poor) > θ*(rich)`.
