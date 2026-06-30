# Krueger–Mitman–Perri (2016) — Aiyagari GE with Unemployment Insurance

A general-equilibrium incomplete-markets economy (Krueger–Mitman–Perri,
Handbook of Macroeconomics 2016): an Aiyagari production economy whose
idiosyncratic risk is **employment risk**, with a flat-tax-financed
**unemployment-insurance** scheme. Built entirely from existing library stages —
the household block is the `examples/unemployment_insurance` device, closed in
general equilibrium by the `examples/aiyagari` tatonnement.

## Household block — existing library stages only

```
EmploymentShock ∘ IncomeReceipt(UI) ∘ ConsumptionSavingsStage
```

| Stage (model role) | Library stage | What it does |
|---|---|---|
| `EmploymentShock` | `MarkovStage` (axis `:employment`) | 2-state employed/unemployed draw `P_e`. |
| `IncomeReceipt(UI)` | `IncomeStage` | `a ↦ (1+r)a + [e·w(1−τ) + (1−e)·ρw]`: employed earn the after-tax wage `w(1−τ)`, unemployed collect the UI benefit `b = ρw`. `r, w, τ, ρ` ride in `env`. |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` | Choose next assets on the `:wealth` grid; budget `c = a_in − a_end`, CRRA. |

No bespoke stage. The UI policy lives in the receipt budget closure.

## What closes the model (the outer loop)

- **General equilibrium on K.** Cobb–Douglas factor prices `r(K), w(K)` with
  effective labor `L = π_e` (the employed share). The fixed point
  `K = ∫ a dΛ(K)` is found by **damped tatonnement** (`steady_state.jl`).
- **Balanced-budget UI.** The employment Markov is exogenous and K-independent,
  so its stationary employment share is constant. Balancing `τ·w·L = ρ·w·π_u`
  pins the flat tax `τ = ρ·π_u/π_e`, **wage-independent** (the wage cancels) and
  hence constant along the tatonnement.

## Equilibrium (default calibration)

```
π_u = 0.0909, π_e = 0.9091, balanced-budget τ = 0.0400
Converged in 8 outer iterations.
K = 4.97,  r = 0.0414  (1/β − 1 = 0.0417),  w = 1.18,  ΣΛ = 1.000000
```

`r` sits just below the time-preference rate `1/β − 1`, the textbook Aiyagari
signature; the UI tax is self-financing.

## Run

```julia
# from HouseholdStages/
julia --project=. examples/krueger_mitman_perri/steady_state.jl
```
