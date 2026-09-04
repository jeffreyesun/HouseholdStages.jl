# Discount-factor heterogeneity — Krusell–Smith / CSTW (2017)

Incomplete-markets economy with a spread of fixed discount-factor types `β`
(Krusell–Smith 1998 stochastic-β; Carroll–Slacalek–Tokuoka–White 2017). The
β-spread is the canonical device for matching the wealth distribution:
patient types build the top tail, impatient types stay near the constraint.

## Household block (pure `⊕` / `product` of exported stages)

Each β-type solves the same canonical chain, differing only in the β fed to
its `ConsumptionSavingsStage`:

```
block_i = MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage(β = β_i)
household = product(block_1, …, block_n; axis = :beta)
```

`β` is a `Float64` field, so all per-type blocks are one chain at different
parameter values, sharing the start and end layouts `product` asks of its
factors. The
`:beta` axis is a size-1 **singleton** in the block layout; `product` grows it
`1 → n`. The direct sum is block-diagonal (a household keeps its β forever), so
each slice is its own independent stationary problem.

## `product` works with the STANDARD solver — no friction

`define_moments!` wraps the product in a singleton `ChainStage`, so
`solve_steady_state_given_env!(hh, env)` runs directly: VFI to a fixed point
and Λ to stationarity on the fused `(N_w, n_ε, n_β)` tensor, each β-slice
converging independently. Unlike the finite-horizon life-cycle product (which
needs cross-age threading the block-diagonal form does not supply), discount
heterogeneity needs no cross-slice wiring, so **no custom driver is required**.

## Outer layer (example-side)

Partial equilibrium: fixed `r` (set below `1/β_max − 1` so every type is
stationary), `w`; single solve. Per-β wealth split is plain aggregation over
the `:beta` slices of `Λ`.

## Run

```
julia --startup-file=no --project=. examples/discount_heterogeneity/steady_state.jl
```

Result (defaults, β ∈ {0.94, 0.96, 0.98}, r = 0.015): ΣΛ = 1, `V` finite,
`A_mean ≈ 8.58`; per-β mean wealth `0.67 / 1.28 / 23.80` — sharply increasing
in β, the patient type holding the entire tail (the CSTW mechanism).
