# JMP household block — "Continuation Value is All You Need"

The household side of the neural-VFI testbed (the JMP / "Continuation
Value is All You Need" model: a spatial heterogeneous-agent GE model
with aggregate shocks, households choosing **location, housing, and
consumption/savings** under idiosyncratic income and aggregate shocks),
expressed entirely as a composition of stages the library already
ships. This is **not** a port or replacement of the JMP-fresh codebase
— only its within-period **household block**, dogfooded into the
shipped stage vocabulary, plus a small-grid CPU steady-state driver
that solves.

The full model runs `N_K=129, N_Z=5, N_H=7, N_LOC=2447` (~11M points)
on GPU. Here every axis is a handful of points (`N_w=12`, 3 income
states, 3 housing sizes, 2 locations → 216 cells) so the steady state
solves in seconds on CPU.

## The `∘` chain

The within-period timing is read straight off the JMP backward sweep
`iterate_V_backward!` (`code/within-period model/household_problem.jl`),
written here in **time order** (the leftmost stage acts first):

```
hh = price ∘ sell ∘ move ∘ buy ∘ income ∘ savings ∘ shock
```

| stage | library stage | JMP operation (`code/within-period model/stages/…`) |
|---|---|---|
| `price`   | `AssetPriceChangeStage(holdings_axis = :h)` | house revaluation `wealth ↦ wealth + (q−q_last)·h` (`asset_price_change.jl`, `get_V_price`). Identity in steady state (`q_last = q`); present for structural fidelity — it carries the aggregate-shock channel in the full model. |
| `sell`    | `SellHomeStage(axis = :h)` | keep-vs-sell on the housing axis (`sell_home.jl`, `get_V_choosesell`). |
| `move`    | `MigrationStage(axis = :location)` | Gumbel/logit location choice with a pairwise move-cost matrix and taste-shock scale `ε` (`move.jl`, `get_V_move`). |
| `buy`     | `BuyHomeStage(axis = :h)` | homebuying — renter may buy any size, owner gated to keep its size (`buy_home.jl`, `get_V_choosebuy` / `get_P_buy`). |
| `income`  | `WealthChangeStage(axis = :wealth)` | the JMP budget `get_income`: interest on net bondholdings `(wealth−q·h)` at `r` if ≥0 else `r_m`, plus labour `A·z` and net rental `ρ·h − (ρ·χ+δ)·h` (`income.jl`, `household_problem.jl::get_income`). |
| `savings` | `ConsumptionSavingsStage(axis = :wealth)` | choose next total wealth, `c = cash_on_hand − wealth'`; flow utility is the JMP log-CES indirect utility `log(c / P(ρ))` (`consumption_savings.jl`, `household_problem.jl::get_indirect_u`). |
| `shock`   | `MarkovStage(axis = :income)` | idiosyncratic income (z) Markov transition at period end (`income_shock.jl`, `get_V_preshock`). |

Wealth is **total** wealth (it includes the house value `q·h`); the
mortgage is implicit, as negative net bondholdings `wealth − q·h < 0`
charged at `r_m`, exactly as in the JMP `get_interest`. The two
locations differ in productivity `A` and rent `ρ`.

No `@definestage`, no `struct <: AbstractStage`, no custom
kernel/`backward!`/`forward!`. Everything outside the block (the env /
exogenous prices) is plain driver code; the GE tatonnement and the
aggregate-shock loop that close the full model are not built here (a
fixed-aggregate steady state is sufficient to demonstrate the block).

## What did NOT map cleanly

One JMP household operation is **not** expressed: the **realtor fee /
sale proceeds** `wealth ↦ wealth − ϕ·q·h` that `sell_home.jl` applies
to sellers (`get_wealth_postsell`). The library's `SellHomeStage`
deliberately does not apply it and documents why (the stage's own
`#TODO`): sellers and pre-existing renters both land on the renter
slice after the choice, so a *following* `WealthChangeStage` cannot
tell them apart to charge the fee — the seller's pre-sale size has
been overwritten on the `:h` axis. Charging `ϕ·q·h` on the post-sell
cell gives `ϕ·q·0 = 0`. Expressing it would need a pre-sell-size axis
that no shipped stage populates. This example sets `ϕ = 0` (no sale
friction). Everything else in the JMP household block — the seven-stage
V/Λ chain, the migration logit, the gated buy/sell choices, the
total-wealth budget with implicit mortgage, the log-CES consumption
problem — composes from existing stages.

**Verdict:** the JMP household side is essentially fully expressible by
composition; the single exception is a *wealth credit attached to the
sell choice*, which is a known limitation of the gated-choice stages
(it needs a carried pre-sell size, not a new household stage).

## Run it

```
julia --project=. examples/jmp_fresh/steady_state.jl
```

Solves to a stationary measure (`mass(Λ) ≈ 1`, finite `V`): ~84%
ownership, both locations populated (~51% home), at the default
calibration.
