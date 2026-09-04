# Costly diffusion — deliberate variance-INCREASE (the θ↑ dual)

The **"negative" of rational inattention**: a household that pays to *add*
dispersion to its next-period wealth — a deliberate mean-preserving spread —
rather than to sharpen it. This realizes the θ↑ ("diffuse") reading of
`MeanPreservingSpreadStage` flagged in `MODEL_CATALOG.md` §7 (the one ◐ on the
opposites table) as a **shipped, solved** example, while staying a pure
composition of existing library stages.

The engine is the **same value-function convexity** that drives Vereshchagina–
Hopenhayn risk-shifting (`examples/risk_shifting`): a limited-liability wealth
floor `max(wealth, a_floor)` makes the continuation locally **convex** just
above the floor (the downside is absorbed by the floor, the upside is kept). A
mean-preserving spread on a convex region raises `E[V]`, so a household near the
floor deliberately **diffuses** — picks θ↑ — even paying a small dispersion
cost. Far from the floor V is concave and the household picks θ = 0.

The point of this example: the entire within-period problem is **five existing
library stages**, in time order, with **no bespoke household stage**.

## Household block (existing stages only)

| Stage (this example) | Library stage | Role |
|---|---|---|
| `IncomeShock` | `MarkovStage` (axis `:income`) | Persistent earnings shock — spreads the stationary distribution. |
| `Receipt` | `IncomeStage` `b ↦ (1+r)b + w·y` | Cash-on-hand each period; impatience keeps wealth bounded, the injection refills it. |
| `Savings` | `ConsumptionSavingsStage` | Choose next-period wealth `b'`; `c = x − b'`, CRRA. |
| `Diffuse` | `MeanPreservingSpreadStage` (axis `:wealth`) | Pick the continuous dispersion `θ ∈ [0, θ_max]` of a Gaussian mean-preserving spread of `b'` (sd `θ`, clamped) at cost `c(θ) = λ·θ²`. **θ = 0 is free; more spread costs more** — the costly-diffusion cost direction. |
| `LimitedLiability` | `WealthChangeStage` `b ↦ max(b, a_floor)` | Limited liability: a bad spread draw cannot push wealth below `a_floor`. This floor **convexifies V** just above `a_floor` — the engine that makes deliberate diffusion (θ↑) pay. |

Composed as `IncomeShock ∘ Receipt ∘ Savings ∘ Diffuse ∘ LimitedLiability`.
Backward order reads right-to-left: limited liability floors the continuation,
the diffusion stage sees that floored (convex) V and seats θ*(x), savings picks
`b'`, receipt relabels cash, the Markov stage takes the income expectation.

## Why this is the dual, not a rebuild of two existing examples

- **`risk_shifting`** gambles via `GaussianLoadingStage` — a *multiplicative*
  risky **share**: Gaussian rows at mean `a'·(R_f + θμ)`, sd `|a'|·θ·σ`, over the
  `GaussianLoadingKernel`. Here the lever is `MeanPreservingSpreadStage` — an
  *additive* Gaussian mean-preserving spread of `b'` (mean `b'`, sd `θ`) over the
  `MeanPreservingSpreadKernel` row. **Same row family, different subspace.**
- **`rational_inattention` / `mackowiak_wiederholt`** use the *same* stage
  (`MeanPreservingSpreadStage`) but in the θ↓ **"sharpen"** reading: the dispersion is
  an unwanted byproduct of a noisy signal, θ = 0 is the perfect-attention
  benchmark, and θ* is interior only because of the borrowing constraint. Here
  θ↑ is the **deliberate** choice — the household wants the spread for its own
  sake (convex-V option value), the opposite economic direction.

So this is `MeanPreservingSpreadStage` exercised on the **opposite side** of the
continuation's curvature from the RI examples — the §7 sign-flip made concrete.

## Equilibrium

Returns and income are **exogenous** (partial equilibrium): there is no market
to clear, so the "outer loop" is a single `solve_steady_state_given_env!`. The
limited-liability floor plus the income injection deliver a stationary wealth
distribution with mass churning near the floor (where diffusion is active).

## Parameters (defaults)

`β = 0.94`, `σ = 3` (CRRA), `w = 0.50` income scale; 2-state income
`y ∈ {0.7, 1.3}`; continuous dispersion `θ ∈ [0, 2.0]` (Gaussian spread, sd `θ`);
dispersion cost `c(θ) = λ·θ²`, `λ = 0.02`; floor
`a_floor = 0.50`; wealth grid `N_w = 120`, log-spaced on `[0, 12]`.

## Expected output

```
Costly-diffusion steady state (σ = 3.0, a_floor = 0.50, λ = 0.0200)
  mass(Λ)              = 1.000000
  mean wealth          = 1.1754
  dispersion θ*: poor  = 1.494  (near floor a≈0.50, convex V)
  dispersion θ*: rich  = 0.004  (top decile a≈12.0, concave V)
  mean θ*              = 0.477,  frac(θ*>0) = 0.458
  ⇒ deliberate diffusion: poor spread 366.0× more than rich

Comparative static — mean chosen dispersion vs. dispersion cost λ:
  λ = 0.0000  →  mean θ* = 0.6506,  θ*(poor) = 2.000,  frac(θ*>0) = 0.554
  λ = 0.0050  →  mean θ* = 0.5787,  θ*(poor) = 2.000,  frac(θ*>0) = 0.508
  λ = 0.0200  →  mean θ* = 0.4774,  θ*(poor) = 1.494,  frac(θ*>0) = 0.458
  λ = 0.0800  →  mean θ* = 0.1777,  θ*(poor) = 0.347,  frac(θ*>0) = 0.396
```

The seated policy `θ*(x)` is high near the floor (deliberate diffusion in the
convex region) and zero for the well-capitalized — the mirror image of the
risk-shifting comparative static, and mean θ* falls monotonically as the
dispersion cost λ rises.

## How to run

```julia
julia --project=. examples/costly_diffusion/steady_state.jl
```

Or interactively:

```julia
include("examples/costly_diffusion/steady_state.jl")
costly_diffusion_steady_state()        # full report + (V, Λ, moment, policy)
```
