# Auclert (2019) revaluation / Fisher channel

The **revaluation (balance-sheet) channel** of Auclert (2019, "Monetary Policy
and the Redistribution Channel"): when the price `q` of an existing asset
moves, holders' net worth is revalued by `(q − q_last)·holdings` *before* any
new decision. That within-period jump is a transfer across the wealth
distribution — the redistributive heart of the channel.

## Household block (pure composition of exported stages)

```
IncomeShock ∘ Revalue ∘ IncomeReceipt ∘ ConsumptionSavings
  = MarkovStage(:income)
  ∘ WealthChangeStage(wealth_post = (; wealth, env) ->
                        wealth + (env.q - env.q_last)*wealth)   # Fisher revaluation
  ∘ IncomeStage(wealth_post = (; wealth, income, env) ->
                  (1 + env.r)*wealth + env.w*income)
  ∘ ConsumptionSavingsStage(β)
```

In **steady state** `q = q_last`, so the revaluation term is identically zero
and the stage is inert — the SS is the plain Bewley/Aiyagari fixed point. The
channel activates out of steady state, at `q ≠ q_last`.

### API-friction note (flagged)

The shipped convenience wrapper `AssetPriceChangeStage(layout;
holdings_axis = :wealth)` is the natural call, but with its default
`wealth_axis = :wealth` it declares the dep-axis tuple `(:wealth, :wealth)`;
downstream `dep_combos` then builds `NamedTuple{(:wealth, :wealth)}` →
**"duplicate field name in NamedTuple"**. The wrapper is usable only when the
revalued holdings sit on an axis *distinct* from wealth (a genuine two-asset
state). For the single-asset balance sheet here (wealth **is** the holdings),
we drop to the `WealthChangeStage` the wrapper is built on, with the identical
revaluation law — still a pure composition of exported stages.

## Outer layer (example-side) — the channel demonstration

`steady_state.jl` solves the SS (revaluation inert), then runs a one-pass
demonstration at a perturbed price `q = q_last·(1 + dq)` using only the public
stage verbs (`backward!`, `forward!`):

1. **Analytic aggregate transfer** `∫ (q − q_last)·wealth dΛ_ss =
   (q − q_last)·A_mean` — the Fisher redistribution to asset holders.
2. **One forward pass** of `Λ_ss` at the perturbed vs unperturbed price,
   comparing next-period mean wealth — isolating the revaluation leg.

The full MIT-shock transition path is outer-loop machinery (not built here).

## Run

```
julia --startup-file=no --project=. examples/auclert_revaluation/steady_state.jl
```

Result (defaults, +5% price move): SS `A_mean ≈ 4.13` with revaluation inert
(mean-wealth check matches exactly); analytic transfer `0.207`; one-pass Δ
next-period mean wealth `0.198`, ratio `0.96`. The < 1 ratio is the
within-period MPC out of the revaluation windfall — the redistribution channel
at work.
