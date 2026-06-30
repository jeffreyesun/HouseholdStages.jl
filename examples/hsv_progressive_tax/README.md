# Heathcote–Storesletten–Violante (2017) — Aiyagari with HSV progressive tax

An Aiyagari (1994) general-equilibrium economy whose only departure from the
textbook model is the fiscal scheme: labour income is taxed by the HSV
(2017) log-linear progressive schedule. Post-tax labour income is

```
T(y) = λ · (w·y)^(1−τ)
```

where `τ ∈ [0,1)` is progressivity (τ = 0 ⇒ flat) and `λ` scales the level.
The case τ = 0, λ = 1 nests the no-tax Aiyagari model.

## The household block

The within-period problem is the canonical three-stage spine, with the HSV
schedule folded into the receipt budget closure:

```
IncomeShock ∘ IncomeReceipt(HSV) ∘ ConsumptionSavingsStage
```

| Chain role | Library stage | What it does |
|---|---|---|
| `IncomeShock` | `MarkovStage` (`axis = :income`) | Idiosyncratic labour-productivity Markov draw; K-operator is `P_y`. |
| `IncomeReceipt(HSV)` | `IncomeStage` (`axis = :wealth`) | Receipt `a ↦ (1+r) a + λ·(w·y)^(1−τ)`; `λ, τ` from `env`. |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` (`axis = :wealth`) | Choose next-period capital; implicit budget `c = a_in − a_end`; CRRA utility. |

No bespoke stage — three existing exported stages composed with `∘`. The
HSV progressivity enters **only** through the receipt closure; the tax
parameters ride in `env` alongside the cleared prices `r, w`. The
`K_supplied = ∫ wealth dΛ` moment is attached for the tatonnement.

## Equilibrium framing

This is a closed **general equilibrium**. Production is Cobb-Douglas with
fixed labour; `r, w` are marginal products at aggregate capital `K`. The
outer loop is the damped tatonnement on `K` copied from `examples/aiyagari`
(`K ← K + speed·(K_supplied − K)`), iterating the inner V/Λ solve until the
capital market clears.

## Result

Default `HSVParams`: β = 0.96, σ = 1.5, α = 0.36, δ = 0.08; HSV tax
**λ = 0.90, τ = 0.181** (HSV's US estimate); three-state productivity;
`N_w = 300` log-spaced wealth points on `[0, 100]`.

```
Converged in 13 outer iterations.
K   = 5.5708
r   = 0.0399
w   = 1.1877
ΣΛ  = 1.000000
```

The tatonnement converges in 13 outer passes despite a large first-pass
overshoot (the `K_init = 5` start gives a high `r`, so the first
`K_supplied` is far off; the 0.05 damping reins it back in).

## Run

```bash
julia --project=. examples/hsv_progressive_tax/steady_state.jl
```

About 8–9 seconds after first compilation.

## Files

- `model.jl` — parameters (incl. `λ, τ`), layout, the three-stage chain
  with the HSV receipt closure, CRRA utility, Cobb-Douglas `hsv_prices`,
  and the `K_supplied` moment.
- `steady_state.jl` — the damped tatonnement on `K` (HSV params in `env`)
  and its report.
