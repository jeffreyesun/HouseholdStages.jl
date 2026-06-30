# Technology / Network Adoption

A household decides whether to **adopt** a technology whose payoff rises with
the economy-wide **adoption share** (a network externality): the more others
adopt, the more valuable adoption is. The discrete adopt choice is a clean
stage; the externality — payoff depending on the cross-sectional adoption
share — is a distribution-dependent equilibrium the caller closes in the
outer loop, exactly like Krusell–Smith's aggregate state.

The household block is **existing library stages only**.

## Household block

Within-period decomposition, in time order:

```
Flow ∘ Discount ∘ AdoptChoice
```

| Stage | Library stage | What it does |
|---|---|---|
| `Flow` | `UtilityStage` | Flow payoff by technology state: `0` if not adopted, `θ·adoption_share` if adopted (the network benefit, share read from `env`). |
| `Discount` | `TimeDiscountingStage` | `V_start = β·V_end`, the contraction. |
| `AdoptChoice` | `LogitChoiceStage` (axis `:technology` ∈ {not, adopted}) | Charges the one-time adoption cost `κ` to switch `not → adopted`; free to stay or abandon. |

So `V(t) = u(t; share) + β·logsumexp_{t'}[ −C[t,t'] + V(t') ]`.

## Equilibrium notes

The driver closes the **network-externality fixed point** `share = mass(adopted)`
in the outer loop. Because the feedback is positive, the model can have
**multiple equilibria** — so the driver solves from a low and a high initial
share to expose any trap vs. high-adoption equilibrium. The indifference
share is `x* ≈ κ(1−β)/θ`.

Headline result (`θ = 1.0`, `κ = 2.0`, `β = 0.85`, `ε = 0.08`, `x* ≈ 0.30`):

```
from LOW  initial share (0.02): converged share = 0.00   (low-adoption trap)
from HIGH initial share (0.98): converged share = 1.00   (high-adoption eqm)
⇒ MULTIPLE equilibria coexist.
```

The textbook coordination outcome: identical fundamentals, two self-fulfilling
adoption levels.

## How to run

```bash
julia --project=. examples/technology_adoption/steady_state.jl
```
