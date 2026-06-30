# Tombe–Zhu (2019) — Migration over a Region×Sector Composite

Workers reallocate across **region-sector cells**, paying a bilateral
migration cost that depends on both the region change and the sector change.
From "Trade, Migration, and Productivity: A Quantitative Analysis of China"
(AER 2019).

The defining feature here: **one discrete choice moves a composite axis**
whose levels are `(region, sector)` pairs — a single `MigrationStage` over
that composite axis, with **no bespoke household stage**.

## Household block

Within-period decomposition, in time order:

```
Migrate ∘ Receipt ∘ ConsumptionSavings
```

| Stage | Library stage | What it does |
|---|---|---|
| `Migrate` | `MigrationStage` (axis `:rs`) | Logit choice over the composite `(region, sector)` axis; cost `M[(r,s)→(r',s')] = region_move·1{r≠r'} + sector_move·1{s≠s'}`. |
| `Receipt` | `WealthChangeStage` (axis `:wealth`) | Cash-on-hand `(1+r)·a + real_wage(region, sector)`. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` (axis `:wealth`) | Saving/consumption on the log-spaced wealth grid. |

State space: `(wealth, rs)`, where `:rs` is `Discrete` over `(region, sector)`
tuples. The composite migration makes the axis ergodic.

## Equilibrium notes

Partial equilibrium: the region×sector real wages (which embed trade through
goods prices) and the spatial measure clear in the trade-and-migration GE
(the caller's outer loop). Single `solve_steady_state_given_env!`.

Headline result (2 regions × 2 sectors, coast-mfg the high-wage cell):

```
Population by (region, sector):
   (coast, ag)     = 0.15      (coast, mfg)    = 0.68
   (interior, ag)  = 0.04      (interior, mfg) = 0.13
```

Mass concentrates in the high-real-wage coast-manufacturing cell, dampened by
the composite migration cost — gradual reallocation across both margins.

## How to run

```bash
julia --project=. examples/tombe_zhu/steady_state.jl
```
