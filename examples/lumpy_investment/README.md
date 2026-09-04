# Lumpy / non-convex investment (Khan–Thomas 2008)

A firm is an **agent block** under the §6 household↔firm dictionary. This example reads
the (S,s) durable-adjustment shape on the capital axis:

| Household reading | Firm reading | Stage |
|---|---|---|
| wealth `b` | capital `k` | the operative (discrete) level axis |
| income shock | productivity shock `z` | `MarkovStage(:z)` |
| (S,s) durable purchase | **lumpy / fixed-cost investment** | keep/adjust `ArgmaxStage` (§5(i)) |
| transaction / menu cost | fixed adjustment cost `F` | the off-diagonal of the reward matrix |

A fixed cost of *changing the capital level* makes investment **lumpy**: a firm sits in
an inaction band and re-tunes its capital only when productivity has drifted far enough
to justify paying `F`. This is the same object as the (S,s) durable in
`examples/durable_housing`, read on the capital axis.

## The block (four existing stages, no bespoke stage)

```
Shock ∘ Profit ∘ Invest ∘ Discount
= MarkovStage(:z) ∘ UtilityStage(z·k^α − (r+δ)·k) ∘ ArgmaxStage(:k; reward M[k',k]) ∘ TimeDiscountingStage(β)
```

The backward sweep reproduces the lumpy-investment Bellman:

```
V(k,z) = z·k^α − (r+δ)·k + max_{k'} [ −F·1{k'≠k} + β·E[V(k',z')|z] ].
```

Two load-bearing decompositions:

- **Profit lives in a `UtilityStage`, not the choice stage.** Operating profit `z·k^α`
  depends on *both* axes (`k` and `z`), but an `ArgmaxStage` reward on the capital axis
  sees only the capital pair `(k', k)`. So the `z`-dependence of the flow must be a
  separate `UtilityStage` — exactly as in `examples/capital_investment`. The
  per-period Jorgensonian user cost `(r+δ)·k` of *holding* the level also goes here, so
  the only adjustment friction left for the choice is the fixed cost `F`.

- **The (S,s) reward is a plain `(after, before)` matrix.** `M[k', k] = −F·1{k' ≠ k}`:
  keeping the level (the diagonal) is free, jumping to any other level pays `F`. A plain
  `Matrix` is the normal `ArgmaxStage` reward parameterization (the same role
  `to_matrix_source` plays for `ConsumptionSavingsStage`/`BuyHomeStage`). The fixed cost
  makes the reward non-supermodular (a monotone solve would not apply; the argmax is
  brute). The capital axis is **discrete** so "keep" (`k' = k`) is an exact
  grid point.

## What is the outer loop (the caller's, never the block)

The rental rate / capital price and the cross-sectional capital distribution as an
aggregate state are partial-equilibrium-exogenous here (a single stationary solve),
exactly as for Aiyagari.

## Running it

```julia
julia --project=. examples/lumpy_investment/steady_state.jl
```

At the default calibration the block solves to a finite value everywhere, mass conserved,
mean capital ≈ 3.6, a non-degenerate capital marginal, capital rising in `z`, and an
**adjustment frequency ≈ 1.9%** — most firms sit in the inaction band each period and a
small minority re-tune in bursts, the signature lumpy-investment pattern.

## Related

The **convex** investment component (Cooper–Haltiwanger smooth `φ(k'−k)²` adjustment) is
`examples/capital_investment`. The generalized-(S,s) smoothing (Caballero–Engel 1999, a
distribution of adjustment hazards) replaces the keep/adjust `ArgmaxStage` with a
`LogitChoiceStage(:k over {keep, adjust}; ε)` — the adjustment-hazard function is the
logit probability.
