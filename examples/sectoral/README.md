# Roy-style Occupational / Sectoral Choice

A worker chooses, each period, which sector (occupation) to work in.
Sectors pay different wages per efficiency unit; idiosyncratic
productivity follows a Markov process common across sectors; switching
sectors costs a utility penalty. This is the dynamic Roy (1951) model with
moving costs, in the spirit of the occupational/trade-mobility literature
(Artuç–Chaudhuri–McLaren 2010, Dix-Carneiro 2014).

The whole household block is assembled from **existing library stages** —
no bespoke household stage, kernel, or per-cell value/transition logic is
defined in this example.

## Household block

Within-period decomposition, in time order:

```
ProductivityShock ∘ SectorSwitch ∘ Earnings ∘ ConsumptionSavings
```

| Stage (this example) | Library stage | What it does |
|---|---|---|
| `ProductivityShock` | `MarkovStage` (axis `:prod`) | Idiosyncratic efficiency-unit draw, common across sectors. |
| `SectorSwitch` | `SectorSwitchingStage` (sugar over `LogitChoiceStage`, axis `:sector`) | Logit choice of next sector with switching cost `κ` (utils) for any move `i → j`, free to stay. |
| `Earnings` | `WealthChangeStage` (deterministic) | Receives the chosen sector's wage `w[sector]` times productivity, plus `(1+r)` on wealth. Reads `w_sector` from `env`. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` (sugar over `ContinuousArgmaxStage`, axis `:wealth`) | Saving/consumption choice on the log-spaced wealth grid. |

State space: `(wealth, prod, sector)`. The sector wage enters earnings
through `WealthChangeStage` reading `w[sector]` from env; the switching
cost enters the logit directly. Comparative advantage (the Roy logic)
lives in the interaction of the sector wage levels with the productivity
draw and the switching cost.

## Equilibrium notes

This is a **partial-equilibrium / fixed-price stationary** model: sector
wages `w_sector` and the wealth return `r` are exogenous parameters. The
solve is a single call to `solve_steady_state_given_env!` (inner VFI to a
fixed point, then the stationary `Λ`). A market-clearing closure on the
wages would be an ordinary outer loop in the style of
`../spatial/steady_state.jl` (tatonnement on a per-sector labor-demand
schedule), but it is not needed to exercise the household block, so it is
omitted.

Headline result at the default calibration
(`w_sector = [0.9, 1.1, 1.0]`, `κ = 0.3`, `ε = 0.5`, `r = 0.03`,
`N_w = 200`):

```
K_supplied = 1.99
employment shares:  :ag = 0.25   :mfg = 0.43   :svc = 0.32
```

The high-wage sector (`:mfg`, w = 1.1) draws the largest share and the
low-wage sector (`:ag`, w = 0.9) the smallest — the Roy comparative-
advantage sort, smoothed by the logit noise and the switching cost.

## How to run

```bash
# Stationary solve + per-sector report
julia --project=. examples/sectoral/steady_state.jl

# Test (asserts mass(Λ) ≈ 1, finite V, sensible shares + comparative advantage)
julia --project=. -e 'using HouseholdStages; include("test/test_example_sectoral.jl")'
```
