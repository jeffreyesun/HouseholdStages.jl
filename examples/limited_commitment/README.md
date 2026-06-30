# Alvarez–Jermann (2000) — Limited Commitment, Decentralized Household

The **decentralized household side** of an Alvarez–Jermann limited-commitment
economy: households trade a single asset subject to a **state-contingent
"not-too-tight" solvency constraint** — a per-income-state lower bound on wealth,
tighter in low-income states (where the default outside option is more
tempting). Given those bounds, the problem is the canonical income-fluctuation
spine with the solvency floor enforced on the savings choice.

## Household block — existing library stages only

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage ∘ SolvencyFloor
```

| Stage (model role) | Library stage | What it does |
|---|---|---|
| `IncomeShock` | `MarkovStage` (axis `:income`) | Persistent income draw `P_y`. |
| `IncomeReceipt` | `IncomeStage` | `a ↦ (1+r)a + w·y`. |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` | Choose next assets; budget `c = a_in − a_end`, CRRA. |
| `SolvencyFloor` | `UtilityStage` | Per-state floor `a_end ≥ B(y)` as a flow penalty: `(; wealth, income, env) -> wealth < env.solvency_bound(income) ? −PEN : 0`. Placed **after** the savings choice, it masks the continuation `ConsumptionSavingsStage` optimises over, so the policy never violates the bound. |

The per-state bound vector `B(y)` rides in `env` (`solvency_bound` maps an income
grid value to its calibrated bound).

## What it shows

```
β = 0.96, σ = 1.5, r = 0.0300;  bounds B(y) = [−0.5, −1.5, −2.5] for y = [0.5, 1.0, 1.5]
ΣΛ = 1.000000,  A_mean = +3.19,  VFI iters = 414
  mean a | low income  = +0.63   (bound −0.50)
  mean a | high income = +1.51   (bound −2.50)
```

The solvency floor is respected **exactly** — zero mass below each per-state
bound. The low-income state presses right up to its tight bound (`min a ≈ −0.48`
vs `−0.50`); higher-income states stay cautious and do not fully use their looser
slack, since income persistence means they might transition down. So the
per-state bounds genuinely shape the stationary distribution.

Because the penalty sits on chosen end-of-period wealth and the income Markov
mixes states, a hard `−∞` bound would collapse the effective limit to the
worst-case `min_y B(y)`; the **finite** penalty retains the state-dependence.

## Scope

**Out of scope:** the planner's promised-utility / Pareto-weight fixed point that
*derives* the not-too-tight bounds `B(y)` from primitives. That outer fixed point
is the caller's; here the bounds are taken as given (calibrated in `env`) and only
the decentralized household block is built.

## Why a `UtilityStage` penalty, not `BorrowingConstraintStage`

`BorrowingConstraintStage` masks with `−Inf`, which **breaks the VFI loop**: its
convergence metric `maximum(abs, V_new .- V)` evaluates to `NaN` at steady `−Inf`
cells (`−Inf − −Inf = NaN`), so iteration stops after 2 passes with an
unconverged V (full diagnosis in `examples/borrowing_constraint/README.md`). A
**finite** penalty `−PEN` is the within-constraints stand-in: large enough to
deter violations (here `PEN = 50` gives zero leakage), finite so the VFI norm
stays well-defined. The grid floor sits well inside the natural limit, so no cell
is an infeasible CRRA trap.

## Run

```julia
# from HouseholdStages/
julia --project=. examples/limited_commitment/steady_state.jl
```
