# Bryan–Morten (2019) — Internal Migration with Selection

Workers choose a location under a fixed + variable moving cost. Because
location wages multiply individual ability, the gain from moving to a
high-wage location rises with ability, so migration **selects on ability**.
From "The Aggregate Productivity Effects of Internal Migration" (JPE 2019).

The household block is **existing library stages only** — no bespoke stage.

## Household block

Within-period decomposition, in time order:

```
Migrate ∘ Flow ∘ Discount
```

| Stage | Library stage | What it does |
|---|---|---|
| `Migrate` | `MigrationStage` (sugar over `LogitChoiceStage`, axis `:location`) | Logit location choice with cost `M[i,j] = fixed + variable·|i−j|` (zero diagonal). |
| `Flow` | `UtilityStage` | Destination flow payoff `wage_j · ability + amenity_j` (read from `env`). The wage×ability complementarity is the selection channel. |
| `Discount` | `TimeDiscountingStage` | `V_start = β·V_end`, the contraction. |

So `V(i, ability) = logsumexp_j[ −M[i,j] + wage_j·ability + amenity_j + β·V(j, ability) ]`.
State space: `(ability, location)`; ability is a fixed type, location ergodic
via migration. A **decreasing location amenity** makes low-ability types
prefer the rural amenity (whose wage gain from the city is small), so the
selection shows up in the stationary stock rather than everyone collapsing
into the highest-wage location.

## Equilibrium notes

Partial equilibrium: location wages and the spatial measure clear in the
regional GE (the caller's outer loop). Single `solve_steady_state_given_env!`.

Headline result (`wage = [0.9, 1.0, 1.2]`, `ability = [0.6, 1.0, 1.6]`,
`amenity = [0.30, 0.15, 0.0]`, `ε = 0.10`):

```
Population by location:   rural = 0.50   town = 0.00   city = 0.50
City population by ability type (each type mass 1/3):
   low = 0.00    mid = 0.17    high = 0.33
```

Clean **positive selection**: high-ability types are in the city, low-ability
stay rural, mid types split — exactly the Bryan–Morten sorting.

## How to run

```bash
julia --project=. examples/bryan_morten/steady_state.jl
```
