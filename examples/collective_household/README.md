# Collective Household — fixed Pareto weight (Chiappori 1988/1992; Mazzocco 2007)

The "agent" is a two-member household that allocates a single shared budget
between its members. Under the collective approach, an efficient household
maximises a Pareto-weighted sum of member utilities. With a **fixed** Pareto
weight `μ`, the household is observationally a single planner, so the
per-period objective is a reshaped felicity over the common consumption `c`.

## Household block (composition of existing stages only)

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
=  MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage
```

The block is the Aiyagari spine. The **only** change is the savings felicity
closure — the Pareto-weighted sum of the two members' utilities over their
shares (`c_A = s·c`, `c_B = (1-s)·c`) of the common consumption:

```julia
utility = (cell, c; env) ->
    μ * u_crra(s * c, Val(σ_A)) + (1 - μ) * u_crra((1 - s) * c, Val(σ_B))
```

Members differ in CRRA curvature (`σ_A = 1.5`, `σ_B = 3.0`), so even at a
fixed weight the household's effective curvature is a member-weighted blend —
the collective content. `u_crra` masks `c ≤ 0` to `-Inf`. No new stage, no
extra `utility_axes`.

## Scope: fixed weight only

This builds **only** the fixed-weight (full-commitment) collective household,
which is a clean felicity reshape of a single planner. The evolving-weight /
limited-commitment version (Mazzocco 2007; Voena 2015; Alvarez–Jermann-style
participation constraints) makes the Pareto weight a *state* that updates when
participation constraints bind. That is not a felicity reshape of a single
planner and is out of scope for this exercise.

## Solve

Partial equilibrium: `r`, `w` fixed and exogenous. One
`solve_steady_state_given_env!`.

```
julia --startup-file=no --project=. examples/collective_household/steady_state.jl
```

Run output (`N_w = 150`): total mass `1.000000`, `V` finite, aggregate wealth
`K = 3.18`, top-cell mass fraction `0`, 422 VFI / 239 Λ iterations.
