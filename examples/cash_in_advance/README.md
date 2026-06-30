# Cash-in-Advance (Clower 1967 / Lucas–Stokey 1987)

Money is the single asset; consumption is capped by the real value of the
money carried in: `c ≤ m / P`.

## Household block (composition of existing stages only)

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
=  MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage
```

The block is the Aiyagari spine. The Clower constraint is imposed **inside**
the savings utility closure by masking infeasible consumption:

```julia
utility = (cell, c; env) -> c > cell.wealth / env.P ? -Inf : u_crra(c, Val(σ))
```

`cell.wealth` is the money carried in (the savings-axis start coordinate), so
`cell.wealth / env.P` is real money on hand. The `-Inf` mask is the same
mechanism the stage uses for `c ≤ 0`, and it lives inside the argmax — so the
cash constraint composes cleanly with **no new stage and no second axis**. (A
standalone constraint on *states* would break the VFI metric; masking the
*choice* does not.)

## Solve

Partial equilibrium: `r`, `w`, `P` fixed and exogenous (`r < 1/β - 1`). One
`solve_steady_state_given_env!`.

```
julia --startup-file=no --project=. examples/cash_in_advance/steady_state.jl
```
