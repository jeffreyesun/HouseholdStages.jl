# Heterogeneous Income Profiles (HIP) — Guvenen

A standard incomplete-markets household with **two** earnings-heterogeneity
sources: a *permanent* lifetime-earnings type `θ` (the HIP "profile
heterogeneity," no transitions) and an AR(1)-style persistent shock `ε` on
top. Earnings are the product `θ·ε`, so the permanent type scales the whole
stochastic profile.

## Household block (pure composition of exported stages)

```
IncomeShock ∘ IncomeReceipt(θ·ε) ∘ ConsumptionSavings
  = MarkovStage(:income)
  ∘ IncomeStage(wealth_post = (; wealth, income, income_type, env) ->
                  (1 + env.r)*wealth + env.w*income_type*income)
  ∘ ConsumptionSavingsStage(β)
```

The permanent type is a plain extra `:income_type` **Discrete** axis with **no
Markov stage** (mass conserved within type). The existing `IncomeStage` budget
closure simply declares `income_type` as an extra kwarg; the field machinery
resolves it against the layout axis. No bespoke stage.

## Outer layer (example-side)

Partial equilibrium: fixed exogenous `r`, `w`, single
`solve_steady_state_given_env!` — no tatonnement. The per-type wealth split is
plain aggregation over the `:income_type` slices of `Λ`.

## Run

```
julia --startup-file=no --project=. examples/heterogeneous_income/steady_state.jl
```

Result (defaults): ΣΛ = 1, `V` finite, `A_mean ≈ 1.53`; mean wealth
`1.18` (θ=0.75) vs `1.88` (θ=1.25), a high/low ratio of `1.59` — the permanent
earnings gap maps into a wealth-distribution gap.
