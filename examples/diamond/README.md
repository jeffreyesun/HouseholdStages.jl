# Diamond (2016) — Skill-Specific City Choice

High- and low-skill workers choose a city, trading off the skill-specific
city wage, the city rent, and the city amenity — where the amenity is
**endogenous to the city's skill composition**. High-skill inflows raise
amenities, which attract more high skill (the sorting feedback). From "The
Determinants and Welfare Implications of US Workers' Diverging Location
Choices by Skill" (AER 2016).

The household block is **existing library stages only**; the endogenous
amenity is closed in the outer loop.

## Household block

Within-period decomposition, in time order:

```
CityChoice ∘ Amenity ∘ Receipt ∘ ConsumptionSavings
```

| Stage | Library stage | What it does |
|---|---|---|
| `CityChoice` | `LogitChoiceStage` (axis `:city`) | Logit city choice with a per-move cost (zero diagonal). |
| `Amenity` | `UtilityStage` | City amenity `A[city]` (read from `env` — the endogenous object). |
| `Receipt` | `WealthChangeStage` (axis `:wealth`) | Cash-on-hand `(1+r)·a + wage(city, skill) − rent(city)`. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` (axis `:wealth`) | Saving/consumption on the log-spaced wealth grid. |

State space: `(wealth, skill, city)`; skill is a fixed type, city ergodic via
the choice.

## Equilibrium notes

The driver closes the **endogenous-amenity fixed point** in the outer loop:

```
A[c] = amenity_base[c] + spillover · (high-skill share in city c)
```

solved by damped iteration; wages and rents are held exogenous (a full GE
would clear those in the same outer loop). This is the Diamond sorting
feedback — the household block never changes, only the driver closes it.

Headline result (3 cities, the `hub` paying a large high-skill premium):

```
Amenity fixed point converges in ~10 iterations.
Endogenous amenities A = [0.185, 0.274, 0.541]
Population by city:        rustbelt = 0.15   sunbelt = 0.17   hub = 0.68
High-skill share by city:  rustbelt = 0.23   sunbelt = 0.28   hub = 0.61
```

The hub draws the high-skilled; their inflow raises its amenity, reinforcing
the sort.

## How to run

```bash
julia --project=. examples/diamond/steady_state.jl
```
