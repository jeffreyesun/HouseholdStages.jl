# Lumpy / (S,s) labor adjustment — Cooper–Haltiwanger–Willis (2007)

A firm with employment level `n` and an idiosyncratic productivity state `z` that
pays a **fixed cost** `F` to change its headcount. Because the cost is paid in full
for any non-zero change, hiring/firing is **lumpy**: the firm sits in an inaction
band, re-tuning `n` only when productivity has drifted far enough to justify the
cost. The within-period firm block is built from **existing library stages only** —
no bespoke stage is rolled here.

## Household ↔ firm dictionary

This is a relabeling of the (S,s) keep/adjust family onto a labor axis:

| Household object | This model |
|---|---|
| wealth / saving `b` | employment level `n` (operative axis) |
| income shock | productivity shock `z` (`MarkovStage`) |
| (S,s) durable purchase | lumpy / fixed-cost hiring–firing (keep/adjust `ArgmaxStage`) |

It is the **same object** as the (S,s) durable in `examples/durable_housing` and the
lumpy capital in `examples/lumpy_investment` (§5(i) of the catalog), read on the
employment axis.

## Firm block (existing stages only)

State space: `(:n, :z)` — employment `n` (a **discrete** grid, so "keep" `n' = n` is
an exact grid point) and idiosyncratic productivity `z` (AR(1)-in-logs, Rouwenhorst-
discretized to `N_z` states).

| Stage (time order) | Library stage | What it does |
|---|---|---|
| `shock`    | `MarkovStage(:z)` | Productivity `z` transitions (persistent AR(1) in logs). |
| `profit`   | `UtilityStage(z·n^θ − w·n)` | Flow profit: revenue `z·n^θ` (DRS, θ<1) net of the wage bill `w·n`. Closure `(; n, z) -> z·n^θ − w·n` reads **both** axes. |
| `adjust`   | `ArgmaxStage(:n; reward = M)` | From `n` pick next headcount `n'`; reward `M[n', n] = −F·1{n' ≠ n}` — keeping is free, any change pays the fixed cost `F`. `search = :brute`. |
| `discount` | `TimeDiscountingStage(β)` | Supplies `β·V_end` before the argmax, `β = 1/(1+r)`. |

Block: **`shock ∘ profit ∘ adjust`** (where `adjust = ArgmaxStage ∘ TimeDiscountingStage`;
`∘` runs the left stage first). Moments: `mean_n`, `mean_profit`, `mean_z`.

Exactly:

```
MarkovStage(:z) ∘ UtilityStage(z·n^θ − w·n) ∘ ArgmaxStage(:n; reward = M[n',n]) ∘ TimeDiscountingStage(β)
```

The backward sweep reproduces the lumpy-labor Bellman

```
V(n,z) = z·n^θ − w·n + max_{n'} [ −F·1{n'≠n} + β·E[V(n',z')|z] ],
```

with frictionless target `n*(z) = (θz/w)^{1/(1−θ)}` (the level a firm would hold each
period absent `F`).

## The design points

**Why profit lives in a separate `UtilityStage`.** The `ArgmaxStage` reward is a
matrix `M[n', n]` over the employment **pair** only — it cannot see `z`. So the
z-dependent flow `z·n^θ − w·n` must live in a separate `UtilityStage` whose closure
reads both axes. Employment then responds to `z` purely through the continuation
value, and the persistent `z` chain spreads stationary mass over `(n, z)`.

**Why the wage bill is in the flow.** `w·n` is the per-period cost of *holding* a
level of employment, paid every period regardless of adjustment. With it in the flow,
holding any level is costly, the per-period payoff is single-peaked in `n` (interior
target `n*(z)`), and the **only** adjustment friction left is the fixed cost `F` —
the textbook (S,s) inaction structure.

**Why `:brute`.** The fixed cost `−F·1{n'≠n}` makes `M` **non-supermodular** (the
diagonal is special), so the monotone/`:divide_conquer` solve does not apply; the
argmax is taken by brute force over the `n'` grid.

## Outer loop (example-side, allowed)

Partial equilibrium: the wage `w` and discount rate `r` are exogenous, so there is no
market to clear. `steady_state.jl` is a single `solve_steady_state_given_env!` over
the joint `(n, z)` state — `V` to its fixed point, `Λ` forward to stationarity — plus
reporting. Closing labor-market clearing for `w` would be an outer fixed point on top,
exactly as in Aiyagari.

The **adjustment frequency** is re-derived from the solved `V` (it is the cleanest
lumpiness diagnostic). With shock-then-adjust timing the argmax sits inside the
z-expectation, so for state `(n, realized z)` the firm picks
`n'* = argmax_{n'} [−F·1{n'≠n} + β·V(n', z)]` and adjusts iff `n'* ≠ n`. The
population facing the choice is post-shock/pre-adjust, `Λ_pre = Λ · P_z` (employment
unchanged, `z` transitioned); `adj_freq = Σ Λ_pre[adjust] / Σ Λ_pre`.

## Expected output

Baseline (`θ = 0.64`, `w = 0.50`, `F = 0.20`, `r = 0.04`, `N_n = 80 ∈ [0.15, 13]`, `N_z = 7`):

```
  frictionless target n*(z): [0.42, 0.70, 1.18, 1.99, 3.34, 5.62, 9.46]
  V finite everywhere   = true
  ΣΛ (mass conserved)   = 1.00000000
  mean n                = 2.3435
  mean profit (z·n^θ−wn)= 0.6606
  n-marginal: 11 / 80 grid points carry mass
  adjustment frequency  = 0.0683  (fraction hiring/firing; rest sit in inaction band)
  mean n by z           = [0.93, 1.00, 1.69, 2.09, 2.82, 4.22, 8.12]
  n rising in z         = true
```

A few percent of firms hire/fire in any period (most sit in the inaction band), the
employment marginal is spread over ~11 interior grid points, and mean employment
rises monotonically in productivity `z` — the textbook lumpy-labor (S,s) cross-section.

Run:
```
julia --startup-file=no --project=. examples/lumpy_labor/steady_state.jl
```
