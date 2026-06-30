# Monte–Redding–Rossi-Hansberg (2018) — Commuting Choice

Residents of a location choose a **workplace** by a gravity/logit rule,
trading the workplace wage against the bilateral commute cost. From
"Commuting, Migration, and Local Employment Elasticities" (AER 2018).

The household block is assembled from **existing library stages only** — no
bespoke household stage, kernel, or per-cell value/transition logic.

## Household block

Within-period decomposition, in time order:

```
Commute ∘ Amenity ∘ Receipt ∘ ConsumptionSavings
```

| Stage | Library stage | What it does |
|---|---|---|
| `Commute` | `LogitChoiceStage` (axis `:workplace`) | Logit choice of workplace; the cost is a `(; residence)` closure giving the commute cost `κ·|residence − j|` to each workplace `j`. The wage tradeoff enters through the destination value, not a kwarg. |
| `Amenity` | `UtilityStage` | Residence-specific flow amenity `B_i` (constant across workplace). |
| `Receipt` | `WealthChangeStage` (axis `:wealth`) | Cash-on-hand `(1+r)·a + wage_workplace − rent_residence`. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` (axis `:wealth`) | Saving/consumption on the log-spaced wealth grid. |

State space: `(wealth, residence, workplace)`. Residence is a fixed type
(the residential margin is MRRH's slower migration choice, omitted here);
workplace is re-chosen each period via the commuting logit.

## Equilibrium notes

Partial equilibrium: wages, rents, and the residential measure clear in the
spatial GE (the caller's outer loop, as always). The solve is a single
`solve_steady_state_given_env!` at fixed `env`.

Headline result (3 locations on a line, `wage = [0.9, 1.2, 1.0]`,
`rent = [0.10, 0.30, 0.15]`, `κ = 0.20`, `ε = 0.30`):

```
Employment by workplace:  west = 0.18   center = 0.58   east = 0.25
```

The high-wage `center` draws the most workers (the gravity pull), with the
commute cost dampening flows from the far locations.

## How to run

```bash
julia --project=. examples/monte_redding_rrh/steady_state.jl
```
