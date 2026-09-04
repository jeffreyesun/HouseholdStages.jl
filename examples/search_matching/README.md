# Search & matching with a tightness externality — steady state

A Mortensen–Pissarides / McCall income-fluctuation problem built **entirely from
existing `HouseholdStages` stages** — no bespoke household stage. A worker chooses
search effort; the job-finding probability depends on aggregate market tightness `θ`
through the matching function (the externality). In general equilibrium `θ` is
pinned by firms' free-entry vacancy posting, computed in the driver.

## The household block

The within-period problem is a four-stage chain, in time order:

```
Separation ∘ Matching ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

The first two legs ship as **one library call**: `SearchMatchingStage` (derived
sugar) expands to exactly `MarkovStage(separation) ∘ MixingStage(job-search lottery)`;
chains flatten, so the chain leaves are unchanged. The cost/policy recipe is
single-homed in the package (`src/stages/derived/search_matching.jl`).

| Stage | Library stage | What it does |
|---|---|---|
| `Separation` | `MarkovStage` (axis `:emp`), via `SearchMatchingStage` | Employed (`emp = 2`) lose their job w.p. `δ` (transition `[1 0; δ 1−δ]`). Runs **first**, so a worker separated this period searches this same period. |
| `Matching` | `MixingStage` (axis `:emp`), via `SearchMatchingStage` | Unemployed (`emp = 1`) **choose their job-finding probability** `p ∈ [0, 1]` directly — the lottery over `K_A =` "search succeeds" (`[0 1; 0 1]`) and `K_B =` "search fails" (identity); the employed rows coincide, so the employed choice is degenerate (`p* = 0`, cost 0). Convex utils cost `c(p) = κ_s·((1−p)log(1−p) + p)`, closed-form argmax `p*(y) = 1 − exp(−y/κ_s)`; the scale `κ_s = χ/(A_match·θ)` reads `θ` from `env` (the sugar's default). |
| `IncomeReceipt` | `WealthChangeStage` | `b ↦ (1+r)·b + income(emp)`: the employed earn wage `w`, the unemployed earn benefit `b_u`. |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` | Pick `b_end` on the wealth grid; implicit budget `c = b_in − b_end`; CRRA utility, divide-and-conquer monotone search. |

The search cost is a **utility** cost (subtracted in the value recursion), as in
McCall search — not a resource drain on the budget. The scale `κ_s = χ/(A_match·θ)`
is calibrated so the marginal probability cost `c′(p) = −κ_s·log(1−p)` equals the
marginal effort disutility `χ·e(p)` at the effort `e(p) = −log(1−p)/(A·θ)` that the
matching technology `p = 1 − exp(−A·e·θ)` requires — so higher tightness means
cheaper search and higher employment. **Timing:** separation runs first, so a worker
who loses their job this period searches in the same period rather than waiting one
period out. The labor axis `:emp` is categorical (`[:unemp, :emp]`); the
wealth axis is a log-spaced grid (dense near the borrowing constraint, where
policies are most nonlinear).

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
across `θ ∈ {0.5, 1, 2, 4}` (employment 0.94 → 1.00, precautionary wealth falling)
and the free-entry GE: `θ* ≈ 6.01`, `q(θ*) ≈ 0.204`, employment ≈ 1.00, with the
free-entry residual driven below `1e-6`. `θ*` is set purely by the firm side — the
free-entry condition does not involve the household block — so household
calibration moves employment and wealth but not the equilibrium tightness.

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
