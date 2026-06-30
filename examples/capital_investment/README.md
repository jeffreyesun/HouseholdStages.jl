# Capital investment with convex adjustment — Cooper–Haltiwanger (2006)

A firm holding capital `k` and an idiosyncratic profitability state `z`, investing
subject to a **convex adjustment cost** (the convex component of the Cooper–
Haltiwanger 2006 lumpy-investment model). The within-period firm block is built
from **existing library stages only** — no bespoke stage is rolled here. This is a
§1 demonstration of the smooth/convex stock-investment family.

## Firm block (existing stages only)

State space: `(:k, :z)` — capital stock `k` (continuous grid) and idiosyncratic
profitability `z` (an AR(1)-in-logs chain, Rouwenhorst-discretized to `N_z` states).

| Stage (time order) | Library stage | What it does |
|---|---|---|
| `shock` | `MarkovStage(:z)` | The idiosyncratic profitability `z` transitions (persistent AR(1) in logs). |
| `profit` | `UtilityStage(z·k^α)` | Adds the flow operating profit `z·k^α` to the value. Closure `(; k, z) -> z·k^α` reads **both** axes. |
| `invest` | `CapitalInvestmentStage(:k)` | From `k` pick next capital `k'`; pay a convex cost `φ·i²` on **gross** investment `i = k' − (1−δ)k`; discount by `β = 1/(1+r)`. Production set to 0 (profit lives in `UtilityStage`). |

Block: **`shock ∘ profit ∘ invest`** (`∘` runs the left stage first). Moments:
`mean_k`, `mean_profit`, `mean_z`.

## The design point: why profit lives in a separate `UtilityStage`

`CapitalInvestmentStage`'s `production`/`effort_cost` closures are `(value; env)` — they see
**only** the operative axis (`k`) and `env`, never a second state axis. So the `z`
dependence of profit **cannot** enter the investment stage's own reward. The clean
compositional route is to put the z-dependent flow in a separate `UtilityStage`
whose closure `(; k, z) -> z·k^α` can read both axes, leaving the investment stage
to carry only the adjustment cost and depreciation. Capital then responds to `z`
purely through the **continuation value**, and the persistent `z` chain spreads the
stationary mass over `(k, z)` — a genuinely non-degenerate cross-section.

## Why depreciation, and why `CapitalInvestmentStage` not `DurableAdjustmentStage`

With concave profit `z·k^α` (`α<1`), no depreciation, and a one-time convex cost,
capital is costless to hold forever once installed — marginal profit `α z k^{α−1}`
is positive for all `k`, so the firm would accumulate without bound (no finite
optimum, `V` unbounded). Depreciation `δ` makes holding capital costly every period
(maintaining `k` needs gross investment `δk` at cost `φ(δk)²`), so the per-period
payoff is eventually decreasing in `k` and the optimum is finite. `CapitalInvestmentStage`
carries `δ` in its gross-investment definition `i = k' − (1−δ)k`;
`DurableAdjustmentStage` (cost on the net change `k'−k`) does not — hence the choice
here. The convex reward `−φ·(k'−(1−δ)k)²` is supermodular in `(k', k)`, so
`CapitalInvestmentStage`'s `:divide_conquer` monotone solve is valid.

## Driver (example-side, allowed)

Partial equilibrium: the discount rate `r` is exogenous, so there is no market to
clear. `steady_state.jl` is a single `solve_steady_state_given_env!` over the joint
`(k, z)` state — `V` to its fixed point, `Λ` forward to stationarity — plus
reporting code (k–z policy profile, marginal spread).

## Expected output

Baseline (`α = 0.70`, `φ = 1.0`, `δ = 0.15`, `r = 0.04`, `N_k = 120`, `N_z = 7`):

```
  V finite everywhere      = true
  ΣΛ (mass conserved)      = 1.00000000
  mean k                   = 7.5199
  mean profit (z·k^α)      = 4.7854
  k-marginal: 29 / 120 grid points carry mass
  k quantiles q10/q50/q90  = 6.474 / 7.470 / 8.665
  k NON-degenerate         = true
  mean k by z              = [6.168, 6.543, 6.973, 7.463, 8.026, 8.691, 9.458]
  k rising in z            = true
```

The capital marginal is spread over ~29 grid points (interior to the grid), and
mean capital rises monotonically in profitability `z` — exactly the convex-
adjustment investment cross-section.

Run:
```
julia --startup-file=no --project=. examples/capital_investment/steady_state.jl
```
