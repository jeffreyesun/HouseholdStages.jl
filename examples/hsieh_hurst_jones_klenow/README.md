# Hsieh–Hurst–Jones–Klenow (2019) — Occupational Choice with Wedges

People of different **groups** sort across occupations under Gumbel
preference shocks, but face group-specific **discriminatory wedges** `τ` that
tax entry into some occupations. The wedges misallocate talent; removing them
reallocates workers and raises output. From "The Allocation of Talent and
U.S. Economic Growth" (Econometrica 2019).

The household block is the catalog's exact composition — **existing library
stages only**.

## Household block

Within-period decomposition, in time order:

```
OccChoice ∘ Flow ∘ Discount
```

| Stage | Library stage | What it does |
|---|---|---|
| `OccChoice` | `LogitChoiceStage` (axis `:occupation`) | Logit occupation choice; the cost is a `(; group)` closure giving the group's wedge `τ[group, j]` to enter occupation `j`. |
| `Flow` | `UtilityStage` | Occupation flow payoff `wage[occupation]` (read from `env`). |
| `Discount` | `TimeDiscountingStage` | `V_start = β·V_end`, the contraction. |

So `V(occ, group) = logsumexp_j[ −τ[group,j] + wage_j + β·V(j, group) ]`.
State space: `(group, occupation)`; group is a fixed type, occupation ergodic.
The group-varying wedge is exactly the `FromEnv`-style heterogeneity the
catalog flags.

(Fidelity note: HHJK's comparative advantage is a Fréchet occupation-talent
draw; here occupations differ by a group-neutral productivity, so the
segregation is driven purely by the wedges. An explicit talent axis would be
one more `MarkovStage`/type axis and leave the block a pure composition.)

## Equilibrium notes

Partial equilibrium: occupation wages clear in the talent-allocation GE (the
caller's outer loop). Single `solve_steady_state_given_env!`.

Headline result (the disadvantaged group taxed on the skilled occupation):

```
Occupation shares within each group (each group mass 1/2):
   skilled : advantaged = 0.43   disadvantaged = 0.11
   home    : advantaged = 0.02   disadvantaged = 0.08
```

The wedge pushes the disadvantaged group out of the high-wage skilled
occupation and into the low-wage home sector — the misallocation channel.

## How to run

```bash
julia --project=. examples/hsieh_hurst_jones_klenow/steady_state.jl
```
