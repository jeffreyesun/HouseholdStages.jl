# Wealth-in-Utility / "Capitalist Spirit" (Carroll 2000; Francis 2009)

Wealth is held not only for the future consumption it buys but for its own
sake. Felicity is `u(c) + v(b')`, where `b'` is the next-period wealth the
household chooses to carry. The direct taste for wealth raises saving relative
to a pure Aiyagari agent and is the textbook route to a thick-tailed wealth
distribution.

## Household block (composition of existing stages only)

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
=  MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage
```

The block is the Aiyagari spine. The **only** change is the savings felicity
closure:

```julia
utility = (cell, c; env) ->
    u_crra(c, Val(σ)) + χ * u_crra(cell.wealth - c, Val(σ_w))
```

`cell.wealth - c` is the chosen next-period wealth `b' = m - c` (post-receipt
cash-on-hand minus consumption). The additive term `χ·u_crra(b', σ_w)` is the
direct taste for wealth. `u_crra` masks `b' ≤ 0` (and `c ≤ 0`) to `-Inf`, so
the corner is guarded automatically. No new stage, no extra `utility_axes`
— wealth lives on the savings axis itself.

## Stationarity

The wealth-in-utility motive pushes the household to save even at a low return,
so `r < 1/β - 1` is necessary but not sufficient to bound wealth: the CRRA
curvature `σ_w > 0` on the wealth term supplies the diminishing marginal taste
that pins a finite stationary target. With `β = 0.96`, `σ_w = 2`, `χ = 0.10`,
`r = 0.02`, the stationary distribution is finite and well inside the grid.

## Solve

Partial equilibrium: `r`, `w` fixed and exogenous. One
`solve_steady_state_given_env!`.

```
julia --startup-file=no --project=. examples/wealth_in_utility/steady_state.jl
```

Run output (`N_w = 150`): total mass `1.000000`, `V` finite, aggregate wealth
`K = 4.04`, top-cell mass fraction `0` (distribution contained by the grid),
398 VFI / 351 Λ iterations.
