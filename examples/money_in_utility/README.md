# Money-in-the-Utility (Sidrauski 1967 / Brock 1974)

Money is the single asset; the real value of end-of-period money balances
enters felicity directly.

## Household block (composition of existing stages only)

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
=  MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage
```

The block is byte-near the Aiyagari spine. The **only** Sidrauski change is
the savings utility closure:

```julia
utility = (cell, c; env) ->
    u_crra(c, Val(σ)) + χ * u_crra((cell.wealth - c) / env.P, Val(σ_m))
```

`cell.wealth - c` is the chosen next-period money holding `m'` (the saving on
the `:wealth` = money axis), and `m'/P` is real balances. That single additive
term is the whole MIU felicity modification — no new stage, no extra
`utility_axes` (money lives on the savings axis itself).

## Solve

Partial equilibrium: `r`, `w`, `P` fixed and exogenous (`r < 1/β - 1` for
stationarity). One `solve_steady_state_given_env!`.

```
julia --startup-file=no --project=. examples/money_in_utility/steady_state.jl
```
