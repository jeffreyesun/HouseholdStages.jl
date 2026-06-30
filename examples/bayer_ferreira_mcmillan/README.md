# Bayer–Ferreira–McMillan (2007) — Neighborhood Sorting

Households of different income types choose a neighborhood via logit, valuing
amenities, neighborhood composition, and price — and **prices clear** each
neighborhood's fixed housing supply. Richer types are less price-sensitive,
so they outbid the poor for high-amenity neighborhoods (income sorting). From
"A Unified Framework for Measuring Preferences for Schools and Neighborhoods"
(JPE 2007).

The household block is **existing library stages only**; the clearing price
vector is closed in the outer loop.

## Household block

Within-period decomposition, in time order:

```
NbhdChoice ∘ Flow ∘ Discount
```

| Stage | Library stage | What it does |
|---|---|---|
| `NbhdChoice` | `LogitChoiceStage` (axis `:neighborhood`) | Logit neighborhood choice with a small per-move cost. |
| `Flow` | `UtilityStage` | `amenity[n] + λ·rich_share[n] − price[n]/income`; price disutility scaled by `1/income` (richer = less sensitive). Price and composition read from `env`. |
| `Discount` | `TimeDiscountingStage` | `V_start = β·V_end`, the contraction. |

State space: `(income_type, neighborhood)`; income type is a fixed type,
neighborhood ergodic.

## Equilibrium notes

The driver closes the **sorting equilibrium** in the outer loop: a price
tatonnement raises a neighborhood's price until its population equals its
housing capacity, and the rich-share composition is updated to its realized
value each pass. This is the BFM equilibrium — the household block never
changes, only the driver closes it.

Headline result (3 neighborhoods, increasing amenity, equal capacity 1/3):

```
Sorting equilibrium converges in ~58 iterations.
Clearing prices (south ≡ 0):   [0.0, 0.34, 1.03]
Population by neighborhood:     0.33 / 0.33 / 0.33  (= capacity)
Rich share by neighborhood:     south = 0.09   midtown = 0.44   heights = 0.97
```

Prices clear capacity, and the rich sort into the high-amenity `heights` —
income segregation produced by heterogeneous price sensitivity.

## How to run

```bash
julia --project=. examples/bayer_ferreira_mcmillan/steady_state.jl
```
