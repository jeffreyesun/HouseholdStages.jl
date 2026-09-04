# R&D / intangible-knowledge investment — convex-cost accumulation

A firm accumulating an intangible **knowledge stock** via R&D, subject to a
**convex R&D cost** and a stochastic demand/productivity shifter. A textbook
smooth-investment example in the §1 stock-investment family. The within-period firm
block is built from **existing library stages only** — no bespoke stage is rolled.

## Firm block (existing stages only)

State space: `(:knowledge, :shock)` — knowledge stock (continuous grid) and a
demand/productivity shifter (AR(1)-in-logs, Rouwenhorst-discretized to `N_s` states).

| Stage (time order) | Library stage | What it does |
|---|---|---|
| `shock` | `MarkovStage(:shock)` | The demand/productivity shifter transitions (persistent AR(1) in logs). |
| `revenue` | `UtilityStage(shock·knowledge^η)` | Adds flow revenue `shock·knowledge^η` to the value. Closure `(; knowledge, shock) -> shock·knowledge^η` reads **both** axes. |
| `do_rnd` | `CapitalInvestmentStage(:knowledge)` | From `knowledge` pick next stock `knowledge'`; pay a convex cost `c_rnd·i^{1/γ}` on **gross** R&D `i = knowledge' − (1−δ_z)knowledge` (`γ<1` ⇒ exponent `1/γ>1`); discount by `β`. Production set to 0 (revenue lives in `UtilityStage`). |

Block: **`shock ∘ revenue ∘ do_rnd`** (`∘` runs the left stage first). Moments:
`mean_knowledge`, `mean_revenue`, `mean_shock`.

## The design point: why revenue lives in a separate `UtilityStage`

`CapitalInvestmentStage`'s `production`/`effort_cost` closures are `(value; env)` — they see
**only** the operative axis (`knowledge`) and `env`, never a second state axis. So
the `shock` dependence of revenue **cannot** enter the R&D stage's own reward. The
clean compositional route puts the shock-dependent flow in a separate `UtilityStage`
whose closure `(; knowledge, shock) -> shock·knowledge^η` reads both axes, leaving
the R&D stage to carry only the cost and knowledge depreciation. R&D then responds
to demand purely through the **continuation value**, and the persistent `shock`
chain spreads the stationary mass over `(knowledge, shock)` — a non-degenerate
cross-section.

## Why depreciation gives a finite optimum

With concave revenue `shock·knowledge^η` (`η<1`) and a one-time convex R&D cost,
undepreciated knowledge would be costless to hold forever and the firm would
accumulate without bound. Knowledge depreciation `δ_z` makes maintaining the stock
costly each period (holding `knowledge` needs gross R&D `δ_z·knowledge` at cost
`c_rnd(δ_z·knowledge)^{1/γ}`), so for `1/γ > η` the per-period payoff is eventually
decreasing and the optimum is finite. `CapitalInvestmentStage` carries `δ_z` in its
gross-investment definition. The convex reward is supermodular in
`(knowledge', knowledge)`, so the optimal policy is monotone — though the stage
does not lean on that, solving by brute per-column max.

## Driver (example-side, allowed)

Partial equilibrium: no prices to clear. `steady_state.jl` is a single
`solve_steady_state_given_env!` over `(knowledge, shock)` — `V` to its fixed point,
`Λ` forward to stationarity — plus reporting code (knowledge–shock policy profile,
marginal spread).

## Expected output

Baseline (`β = 0.96`, `γ = 0.50`, `η = 0.50`, `δ_z = 0.10`, `c_rnd = 2.0`,
`N_k = 120`, `N_s = 7`):

```
  V finite everywhere      = true
  ΣΛ (mass conserved)      = 1.00000000
  mean knowledge           = 4.6451
  mean revenue (s·k^η)     = 2.4674
  knowledge-marginal: 15 / 120 grid points carry mass
  knowledge q10/q50/q90    = 4.273 / 4.604 / 5.101
  knowledge NON-degenerate = true
  mean knowledge by shock  = [4.098, 4.299, 4.480, 4.637, 4.798, 5.030, 5.311]
  knowledge rising in shock= true
```

The knowledge marginal is spread over ~15 grid points (interior), and mean knowledge
rises monotonically in the demand shock — the convex-cost R&D cross-section.

Run:
```
julia --startup-file=no --project=. examples/rnd_investment/steady_state.jl
```
