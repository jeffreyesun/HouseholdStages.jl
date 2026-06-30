# (S,s) inventory management (Khan–Thomas 2007; Kahn–McConnell–Pérez-Quirós 2002)

A firm is an **agent block** under the §6 household↔firm dictionary. This example reads
the (S,s) inaction-band shape on an **inventory** axis — the same keep/adjust object as the
(S,s) durable in `examples/durable_housing` and the (S,s) capital in `examples/lumpy_investment`,
with ONE structural addition: between periods the stock is not inert, it **depletes** as sales
draw it down.

| (S,s) durable reading | (S,s) inventory reading | Stage |
|---|---|---|
| wealth / durable stock | inventory level `i` | the operative (discrete) level axis |
| income shock | demand shock `d` | `MarkovStage(:demand)` |
| transaction / menu cost | fixed reorder cost `F` | the off-diagonal of the reorder reward matrix |
| (no analogue — stock is inert) | **depletion drift** `i ↦ max(i−d, 0)` | `MarkovStage(:inventory; T = (; demand) -> shift)` |

A fixed cost of *placing a reorder* makes restocking **lumpy**: a firm lets its stock draw down as
sales accumulate, sitting in an inaction band, and reorders up to a target `S` only once the stock
has fallen below a trigger `s` — the textbook (S,s) policy (Scarf 1960). The new piece versus the
durable is the **depletion drift**: each period demand `d` realizes, sales `min(i, d)` knock the
stock down, and the firm faces the reorder choice from the *post-depletion* stock.

## The block (five existing stages, no bespoke stage)

```
Demand ∘ Profit ∘ Deplete ∘ Reorder ∘ Discount
= MarkovStage(:demand)
    ∘ UtilityStage(price·sales − h·hold − κ·stockout)
    ∘ MarkovStage(:inventory; transition_matrix = (; demand) -> shift(demand))
    ∘ ArgmaxStage(:inventory; reward M[i', s])
    ∘ TimeDiscountingStage(β)
```

Period timing (one firm): enter with stock `i` (last period's reorder target); demand `d` realizes;
earn `profit(i, d)`; the stock depletes to `s = max(i − d, 0)`; reorder up to `i' ≥ s` (paying
`F + c·Δ`) or keep `s`; discount; carry `i'` into next period. The backward sweep reproduces the
(S,s) inventory Bellman:

```
V(i, d_prev) = E_{d|d_prev}[ price·sales − h·max(i−d,0) − κ·max(d−i,0)
                             + max_{i' ≥ s} ( −F·1{i'≠s} − c·(i'−s) + β·V(i', d) ) ],   sales = min(i,d).
```

Three load-bearing decompositions:

- **The depletion drift is a dep-closure `MarkovStage`, not a bespoke kernel.** Sales knock the stock
  down by a *deterministic* amount that depends on the realized demand, and a deterministic move IS a
  one-hot ROW-stochastic matrix. So the drift is an ordinary `MarkovStage` on `:inventory` whose
  transition is handed as `(; demand) -> shift(demand)` — the same dep-closure-transition contract as
  `examples/skill_depreciation`'s skill drift `(; emp) -> …` and `examples/marriage_capital`'s
  `(; match_capital) -> …`. `shift(d)` sends each stock index to the index nearest `max(grid[i] − d, 0)`.

- **Profit lives in a `UtilityStage`, and precedes the depletion.** The flow depends on *both* the
  pre-depletion stock `i` and the demand `d` (sales, leftover holding, stockout), but the reorder
  `ArgmaxStage` reward sees only the inventory pair `(i', s)`. So the demand-dependence of the flow
  must be a separate `UtilityStage` — and it must read the *pre*-depletion stock, so it sits before
  the depletion move (which overwrites the inventory axis with the leftover `s`).

- **The (S,s) reorder reward is a plain `(after, before)` matrix.** `M[i', s] = 0` on the diagonal
  (keep the post-depletion stock, free), `−(F + c·(grid[i'] − grid[s]))` above it (reorder up: fixed
  cost plus per-unit purchase), `−Inf` below it (reducing the stock is infeasible). A plain `Matrix`
  is the normal `ArgmaxStage` reward parameterization. The fixed cost makes the reward
  non-supermodular, so the stage uses `search = :brute`. The inventory axis is **discrete** so "keep"
  (`i' = s`) is an exact grid point.

## What is the outer loop (the caller's, never the block)

The goods price, the per-unit purchase cost, and the cross-sectional inventory distribution as an
aggregate state are partial-equilibrium-exogenous here (a single stationary solve), exactly as for
Aiyagari / lumpy investment. Clearing those (e.g. a goods market, or an aggregate-demand process) is
the caller's outer loop, never the firm block.

## Running it

```julia
julia --project=. examples/inventory_ss/steady_state.jl
```

At the default calibration the block solves to a finite value everywhere, mass conserved, mean
inventory ≈ 2.9, mean sales ≈ 1.08, a non-degenerate inventory marginal (16 / 49 grid points carry
mass), and a **reorder frequency ≈ 27%** — most firms sit in the inaction band each period letting
the stock deplete, a minority reorder in bursts. The extracted **(S,s) bands** are a target
`S(d) ≈ 4.0–4.5` and a trigger `s(d) ≈ 0.25–1.25` rising in the demand state: a firm in a
higher-demand state reorders earlier (higher trigger) and to a slightly larger target, the signature
(S,s) inventory pattern.

## Related

The non-depleting (S,s) stock (a durable bought once and held) is `examples/durable_housing`; the
(S,s) capital under a fixed adjustment cost is `examples/lumpy_investment`. The generalized-(S,s)
smoothing (Caballero–Engel 1999, a distribution of reorder hazards) would replace the keep/reorder
`ArgmaxStage` with a `LogitChoiceStage` on the inventory axis — the reorder-hazard function is the
logit probability.
