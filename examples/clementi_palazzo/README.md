# Clementi–Palazzo (2016) — entry + convex investment + endogenous exit

A firm is an **agent block** under the §6 household↔firm dictionary; nothing in the
`V`/`Λ` machinery knows household from firm. This example reads the demographic and
consumption-saving stages as firm dynamics, and **combines** two already-validated
blocks into one recursion:

| Household reading | Firm reading | Stage |
|---|---|---|
| wealth `b` | **capital** `k` | the operative continuous axis (`GriddedContinuous`) |
| income / employment shock | **productivity** shock `z` | `MarkovStage(:z)` |
| saving | **investment** `i = k'−(1−δ)k` | `CapitalInvestmentStage(:k)` |
| convex saving / effort cost | **convex capital-adjustment cost** `φ·i²` | the `effort_cost` closure |
| death | firm **exit** | `EndogenousExit` |
| bequest / value of death | **scrap / liquidation value** `resale·k` | the required `bequest` field |
| birth | firm **entry** | `EntryStage` (the additive `Λ += g` source) |
| consumption / flow felicity | operating profit `z·k^α − c_f` | `UtilityStage` |

Clementi–Palazzo's point is that the **interaction** amplifies business-cycle
fluctuations: entrants draw productivity and start small, invest toward their efficient
scale subject to a convex adjustment cost, and exit endogenously when continuation falls
below scrap. Entry, investment, and selective exit are not three separate margins — they
compound.

## The block (five existing stages, no bespoke stage)

State space: `(:k, :z, :exiting)` — capital `k` (continuous grid), productivity `z` (an
AR(1)-in-logs chain, Rouwenhorst-discretized), and the transient `:exiting` axis the exit
composite requires (declared at size 1, grown to 2 internally, collapsed back).

```
Entry ∘ Profit ∘ Exit ∘ Invest ∘ Shock
= EntryStage(g) ∘ UtilityStage(z·k^α − c_f) ∘ EndogenousExit(scrap)
    ∘ CapitalInvestmentStage(:k) ∘ MarkovStage(:z)
```

(`∘` runs the left stage first in the forward sweep.) The backward (value) sweep runs the
chain right-to-left and reproduces the recursion:

```
V(k,z) = z·k^α − c_f
         + max{ scrap(k) , max_{k'}[ −φ·max(k'−(1−δ)k, 0)² + β·E_{z'|z} V(k',z') ] }.
```

`Shock` forms the continuation expectation `E[V(k',z')|z]`; `Invest`
(= `ArgmaxStage ∘ TimeDiscountingStage`) supplies the discount `β = 1/(1+r)` and picks
next capital `k'`, paying the convex cost `φ·i²` on gross investment `i = k'−(1−δ)k`
(disinvestment is free); `Exit` is the optimal-stopping `max(continuation, scrap)` — the
§5(i) keep-vs-stop `ArgmaxStage` the exit composite wraps over `:exiting` — so a firm
liquidates, collecting `scrap(k) = resale·k`, when even its best continuation falls below
scrap; `Profit` adds the operating profit net of the **fixed operating cost** `c_f`;
`Entry` adds the entrant inflow `M·g`.

Forward, entrants are seeded **small** (lowest capital grid point, productivity drawn from
the invariant distribution `ν` of the `z`-chain), this period's incumbents earn profit,
the seated stopping rule drops the low-`(k,z)` firms' mass (including entrant duds that
draw a doomed productivity), survivors invest toward their target capital, and
productivity transitions. **Mass is not conserved** (entry in, exit out); the stationary
firm mass settles at entrant-inflow / exit-rate.

## How this combines the two parent examples

- **The convex-investment block** is `examples/capital_investment`:
  `MarkovStage(:z) ∘ UtilityStage(z·k^α) ∘ CapitalInvestmentStage(:k)`. As there, the
  `z`-dependent operating profit lives in a **separate** `UtilityStage` reading both `k`
  and `z`, because `CapitalInvestmentStage`'s `(value; env)` closures see only the
  operative axis `k` — this split is what gives the capital stock a non-degenerate
  cross-section over `(k, z)`. Depreciation `δ` (and the convex cost) is what keeps the
  optimal capital finite under concave profit.
- **The entry/exit composite** is `examples/hopenhayn`:
  `EntryStage(g) ∘ … ∘ EndogenousExit(scrap)` on a layout carrying `:exiting`. As there,
  `bequest` (the scrap value) is **required**, mass is non-conserved, and the stationary
  mass is the entry/exit balance.

The **fixed operating cost `c_f`** is load-bearing for exit: without it, low-productivity
firms' continuation never falls below scrap (with a persistent, mean-reverting `z` and a
high `β`, the option value of riding out a bad streak dominates a small scrap), and nobody
exits. The calibration pairs `c_f` with **high persistence and wide productivity
dispersion** (`ρ = 0.95`, `σ = 0.40`) so that a low draw is a long, deep loss-making trap:
the bottom productivity states' continuation falls below scrap and those firms liquidate.

## What `steady_state.jl` reports

A single `solve_steady_state_given_env!` over `(k, z, exiting)` at given prices (partial
equilibrium), then: `V` finiteness; total firm mass; the **exit rate** (mass-weighted
fraction of the pre-exit cross-section whose best continuation `C(k,z)` falls below
`scrap(k)`, with the exact mass-balance cross-check `entry-flow = exit-flow`); mean
capital; mass-weighted **mean `z` versus the entrant mean** (survivor selection — culling
low-`z` firms lifts the survivor mean above the entrant mean); the spread of the
`k`-marginal; and that capital rises in `z`.

Representative numbers at the shipped calibration (`α = 0.70`, `φ = 1.0`, `δ = 0.15`,
`r = 0.10`, `c_f = 3.0`, `scrap = 0.6·k`, `ρ = 0.95`, `σ = 0.40`, `M = 1`):
firm mass ≈ 42.8, exit rate ≈ 2.3% (`exit-flow = entry-flow = 1.0000` to four places),
mean `k` ≈ 11.5 (`q10/q50/q90 ≈ 5.5 / 10.5 / 21.6`, 106/120 grid points occupied),
survivor mean `z` ≈ 3.28 vs entrant mean ≈ 2.19, capital monotone in `z`.

## What is outer loop (the caller's, never the block)

Exactly as for Aiyagari and Hopenhayn: the **free-entry condition** `∑ g·V = c_e` (a
scalar root in the price level — `free_entry_residual` in `model.jl` is the object to zero
on), **aggregate market clearing** (which pins the entrant mass `M`), and the **firm-size
(capital) distribution** as an aggregate state are all the caller's outer loop. Here the
block is solved at given prices; no market is cleared inside it.

## Files

- `model.jl` — parameters, the Rouwenhorst `z`-process, the entrant distribution, the
  firm block `entry ∘ profit ∘ exit ∘ invest ∘ shock`, and `free_entry_residual`.
- `steady_state.jl` — the stationary solve and the diagnostics above (including the
  `best_continuation` reconstruction of the value the exit stage compares against scrap).

## Literature

Clementi & Palazzo (2016, *AEJ: Macroeconomics*), "Entry, Exit, Firm Dynamics, and
Aggregate Fluctuations"; builds on Hopenhayn (1992) entry/exit and the Cooper–Haltiwanger
(2006) convex adjustment cost.
