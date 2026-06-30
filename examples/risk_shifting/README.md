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
| `Gamble` | `MeanVarianceStage` (axis `:wealth`) | Pick project risk `θ ∈ shares`: stake → `a'·(R_f + θ·(R_k − R_f))`, two-point `R_k` (succeed/fail). **Mean-neutral** (`E[R_k] ≈ R_f`) so convexity, not a premium, drives gambling. |
| `LimitedLiability` | `WealthChangeStage` `a ↦ max(z·a, a_floor)` | Limited liability / outside option: a failed gamble cannot push wealth below `a_floor`. This floor **convexifies V** just above `a_floor` — the engine of risk-shifting. |

Composed as `OccShock ∘ Receipt ∘ Savings ∘ Gamble ∘ LimitedLiability`.

### Why mean-variance suffices (no skewness primitive)

The V–H risk-shifting result is driven by **value-function convexity** induced
by the limited-liability floor, interacting with a **mean-variance** project-risk
choice. The canonical V–H gamble is a *two-point* project (succeed big / fail to
the floor) — exactly a two-shock `MeanVarianceStage`. The right-tail emphasis is
not a primitive skewness control; it emerges *endogenously* from the floored
continuation. So the model needs no `KernelChoiceStage` and no new streaming
skewness primitive — `MeanVarianceStage` + a `WealthChangeStage` floor is the
faithful statement.

### Why no occupation axis

The V–H outside option is, mechanically, a **lower bound on the continuation**.
We state it as a lower bound on *realized wealth* (`max(z·a, a_floor)`), which a
`WealthChangeStage` expresses directly and library-only. A literal occupation
axis carrying a flat worker value `W` and an `ArgmaxStage` for `max(V_continue, W)`
would require the savings stage to apply on the entrepreneur leg only — per-axis
gating the library deliberately does not offer without a bespoke kernel. The
wealth-floor formulation is the gating-free statement of the same convexity.

## Equilibrium

Returns, productivity, and the wage are **exogenous** (partial equilibrium):
there is no market to clear, so the "outer loop" is a single
`solve_steady_state_given_env!`. The limited-liability floor plus the cash
injection deliver a stationary wealth distribution (bimodal: a mass of churning
poor entrepreneurs pinned at the floor, and a separated mass of accumulated rich
ones).

## Parameters (defaults)

`β = 0.96`, `σ = 2` (CRRA), `w = 0.40` cash injection; 2-state productivity
`z ∈ {0.98, 1.02}`; project `R_f = 1.06` (safe), risky leg `(R_up, R_dn) =
(1.80, 0.60)` with `p_up = 0.40` ⇒ `E[R_k] = 1.08 ≈ R_f` (mean-neutral); risk
grid `θ ∈ {0, 0.1, …, 1.0}`; floor `a_floor = 0.50`; wealth grid `N_a = 200`,
log-spaced on `[0, 200]`.

## Expected output

```
Risk-shifting steady state (σ = 2.0, a_floor = 0.50, E[R_k] = 1.080 vs R_f = 1.06)
  mass(Λ)               = 1.000000
  mean wealth           = 65.64
  project risk θ*: poor = 0.600  (near floor a≈0.50)
  project risk θ*: rich = 0.000  (top decile a≈200)
  ⇒ risk-shifting: poor gamble 600× more than rich
```

The seated policy `θ*(x)` is high near the floor (gambling for resurrection) and
zero for the well-capitalized — the V–H comparative static.

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
