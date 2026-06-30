# Temptation & Self-Control (Gul–Pesendorfer 2001)

The within-period ranking of a feasible plan is

```
u(c) + w(c) − max_{c̃ feasible} w(c̃)
```

`u` is the commitment (long-run) utility, `w` the temptation utility; the
bracket `w(c) − max_{c̃} w(c̃) ≤ 0` is the self-control cost of resisting the
most tempting feasible option.

## Why this is a clean ✅ (corner temptation = closed-form constant shift)

If the temptation utility `w` is **monotone increasing** in consumption, the
most-tempting feasible option is to consume everything this period — i.e.
drive next-period wealth to the borrowing floor `b_min`. With post-income
cash-on-hand `m = cell.wealth`, the corner consumption is `c̃* = m − b_min`,
so the temptation peak is the **closed form**

```
max_{c̃} w(c̃) = w(m − b_min).
```

This depends only on the cell's cash-on-hand `m`, **not** on the savings
choice `c`. As a function of the choice it is therefore a **constant additive
shift**: it lowers the within-period value `V` (the self-control cost is real)
*without* distorting the policy that maximises `u(c) + w(c)`. That is exactly
the Gul–Pesendorfer solution, and it is time-consistent.

The model catalog flags general temptation as ◐ because the temptation `max`
ranges over the *same* feasible set as the choice (a shared-feasible-set
coupling). **That caveat dissolves under monotone `w`**: the max collapses to
the single corner `b' = b_min`, a closed form in the origin cash-on-hand, so
there is nothing to couple — it folds into one felicity closure.

## Household block (composition of existing stages only)

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
=  MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage
```

Receipt is placed **before** savings (as in Aiyagari), so inside the savings
closure `cell.wealth` is post-receipt cash-on-hand `m = (1+r)b + w·y`. The
felicity closure is the GP ranking with the corner peak folded in as a
constant:

```julia
b_min = w_min   # borrowing floor = grid floor
utility = (cell, c; env) ->
    u_crra(c, Val(σ)) + λ * u_crra(c, Val(σ_t)) - λ * u_crra(cell.wealth - b_min, Val(σ_t))
```

No new stage, no extra `utility_axes`. `u_crra` masks `c ≤ 0` to `-Inf`; the
constant peak term is always finite because post-receipt `m > b_min`.

## Effect: temptation reduces saving

With `w` and `u` sharing CRRA curvature, the policy maximises
`(1+λ)·u(c) + βEV(b')`: the extra weight `λ` on current felicity makes the
household effectively more impatient, so saving falls. Sweeping `λ` at the same
env (`r = 0.02`) confirms the monotone GP direction:

| `λ`  | aggregate wealth `K` |
|------|----------------------|
| 0.00 | 2.41 (Aiyagari baseline) |
| 0.15 | 1.04 (default)       |
| 0.50 | 0.02 (near hand-to-mouth) |

## Solve

Partial equilibrium: `r`, `w` fixed and exogenous. One
`solve_steady_state_given_env!`.

```
julia --startup-file=no --project=. examples/temptation/steady_state.jl
```

Run output (`N_w = 150`, `λ = 0.15`): total mass `1.000000`, `V` finite,
aggregate wealth `K = 1.04`, top-cell mass fraction `0`, 400 VFI / 192 Λ
iterations.
