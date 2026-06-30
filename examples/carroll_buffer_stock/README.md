# Carroll (1997) buffer-stock saving

Impatient consumers with **permanent** (unit-root) and **transitory** income
shocks accumulate a target buffer stock. Catalog §0 ◐: the unit-root permanent
component is not a stationary `MarkovStage` on a fixed grid — Carroll's
**normalization by permanent income** is what makes it expressible.

## Household block (existing stages only)

In ratio-to-permanent-income units (state = normalized cash-on-hand `m`):

```
MarkovStage(:psi) ∘ PointwiseScaleStage((Gψ)^{1-σ} | 1) ∘ MarkovStage(:xi)
  ∘ WealthChangeStage(m = R·a/(Gψ) + ξ) ∘ ConsumptionSavingsStage(:wealth)
```

The normalization leaves exactly two artefacts, both covered by existing stages:

1. **budget division by `G·ψ`** — folded into the `WealthChangeStage` receipt
   closure (reads the `:psi` axis);
2. **continuation reweight `(Gψ)^{1-σ}`** — the CRRA artefact. It varies across
   the permanent-shock axis and must scale **V (backward)** but not **Λ
   (forward)**. That is precisely the asymmetric two-sided
   `PointwiseScaleStage(backward = (Gψ)^{1-σ}, forward = 1)`, with the scale a
   full-layout array (varying only along `:psi`) supplied via `env`.

Permanent and transitory shocks are iid ⇒ degenerate Markov chains with
identical rows. No bespoke stage.

## Key finding

The CRRA permanent-reweight `(Gψ)^{1-σ}` is the one part beyond the plain spine,
and it lands **exactly** on `PointwiseScaleStage`'s asymmetric (backward ≠
forward) capability with an env-supplied array scale. At σ = 1 (log utility) the
reweight is identically 1 and the block collapses to the bare spine.

## Run

```
julia --startup-file=no --project=. examples/carroll_buffer_stock/steady_state.jl
```

Solves to a stationary normalized distribution: `mass(Λ) = 1.0`, mean normalized
cash-on-hand `m ≈ 1.50` (the buffer-stock target), ~333 VFI iters. `mass = 1.0`
confirms the reweight hit V but not Λ.
