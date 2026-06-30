# Search & matching with a tightness externality — steady state

A Mortensen–Pissarides / McCall income-fluctuation problem built **entirely from
existing `HouseholdStages` stages** — no bespoke household stage. A worker chooses
search effort; the job-finding probability depends on aggregate market tightness `θ`
through the matching function (the externality). In general equilibrium `θ` is
pinned by firms' free-entry vacancy posting, computed in the driver.

## The household block

The within-period problem is a three-stage chain, in time order:

```
SearchMatchingStage ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

| Stage | Library stage | What it does |
|---|---|---|
| `SearchMatchingStage` | `SearchMatchingStage` | Unemployed (`emp = 1`) choose search effort `e` from an internal grid; effort costs `cost(e) = χe²/2` utils and finds a job w.p. `p(e, θ) = 1 − exp(−A·e·θ)`. Employed (`emp = 2`) separate at rate `δ`. Backward maxes over effort and stores the effort policy; forward replays the θ-dependent matching row + separation. `θ` rides `FromEnv(:θ)`. |
| `IncomeReceipt` | `WealthChangeStage` | `b ↦ (1+r)·b + income(emp)`: the employed earn wage `w`, the unemployed earn benefit `b_u`. |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` | Pick `b_end` on the wealth grid; implicit budget `c = b_in − b_end`; CRRA utility, divide-and-conquer monotone search. |

The effort cost is a **utility** cost (subtracted in the value recursion), as in
McCall search — not a resource drain on the budget. The labor axis `:emp` is
categorical (`[:unemp, :emp]`); the wealth axis is a log-spaced grid (dense near the
borrowing constraint, where policies are most nonlinear).

No new stage, kernel, or per-cell household value/transition function is defined.
The firm side and tightness closure are plain outer-loop arithmetic.

## Equilibrium

Two drivers in `steady_state.jl`, each rolling its own outer loop (the library
supplies only the per-env V/Λ fixed-point solve):

1. **Partial equilibrium** — fix `θ` exogenously, one inner solve. Exercises the
   stage's `FromEnv(:θ)` contract directly. A tighter market raises the job-finding
   rate, so employment rises and precautionary savings fall (workers self-insure
   less when jobs are easy to find).

2. **Free-entry general equilibrium** — close `θ` with the firm's vacancy-posting
   condition `κ = q(θ)·J`, where
   - `q(θ) = A·θ^(−η)` is the vacancy-filling rate (falling in `θ`), and
   - `J = (z − w) / (1 − β(1−δ))` is the value of a filled job — a closed-form
     geometric sum (discounted flow surplus `z − w` over the match's expected life),
     **not** a firm Bellman iteration.

   The free-entry residual `κ − q(θ)·J` is a scalar function of `θ`; bisection finds
   its root. The household block is solved once at the converged `θ*`.

## Running

```julia
julia --project=. examples/search_matching/steady_state.jl
```

At the baseline calibration the driver reports the PE employment/wealth trade-off
across `θ ∈ {0.5, 1, 2, 4}` and the free-entry GE: `θ* ≈ 6.01`, `q(θ*) ≈ 0.204`,
employment ≈ 0.90, with the free-entry residual driven below `1e-6`.

## Parameters (baseline)

| | | | |
|---|---|---|---|
| `β = 0.96` | `σ = 2.0` | `r = 0.03` | `w = 1.0` |
| `b_u = 0.4` | `δ = 0.10` | `χ = 0.5` | `A_match = 0.5` |
| `z = 1.2` | `κ = 0.30` | `η = 0.5` | `N_w = 120` on `[0, 40]` (log) |

## References

- McCall (1970), "Economics of Information and Job Search," *QJE*.
- Mortensen & Pissarides (1994), "Job Creation and Job Destruction," *ReStud*.
- Pissarides (2000), *Equilibrium Unemployment Theory*, 2nd ed.
- Krusell, Mukoyama & Şahin (2010), "Labour-Market Matching with Precautionary
  Savings," *ReStud* (the search + savings embedding).
