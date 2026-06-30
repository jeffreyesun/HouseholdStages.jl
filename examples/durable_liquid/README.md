# Durable + liquid asset, lumpy (S,s) adjustment (Berger–Vavra)

A heterogeneous-agent household holding a **lumpy durable** `d` (a few discrete
sizes) alongside a **liquid asset** `b`, in the (S,s) tradition of
**Berger–Vavra (2015)** and **Díaz–Luengo-Prado (2010)**. The point of this
example: the within-period problem is a composition of **existing library
stages** — no bespoke household stage. The hard part is that changing the
durable is a single choice that moves **two axes at once**: it sets the durable
stock *and* debits the liquid balance by the down-payment (the purchase price of
the net stock change plus a fixed adjustment cost). That cross-axis coupling is
exactly what halts naive attempts; it is expressible via the
**auxiliary-choice-axis pattern** (Route A; cf. `examples/two_asset_hank` and
`examples/habit`).

## Household block

The within-period problem, in time order:

```
Depreciate ∘ IncomeShock ∘ Receipt ∘ [ ChooseD' ∘ Debit ∘ SetDurable ∘ Forget ] ∘ ConsumptionSavings
```

| Stage | Library stage | What it does |
|---|---|---|
| `Depreciate` | `MarkovStage` (`:durable`) | W.p. `π_dep` the stock drops one level (breakdown/depreciation); level 1 absorbing. The churn source that keeps the (S,s) adjustment margin live. |
| `IncomeShock` | `MarkovStage` (`:income`) | 2-state labour-income Markov. |
| `Receipt` | `WealthChangeStage` (`:liquid`) | Cash on hand `b ↦ (1+r_b)·b + w·y`. |
| `ChooseD'` | `ArgmaxStage` (`:durable_choice`) | Picks the durable target `d'` onto a separate auxiliary axis (reward `0`; `:brute`, since the lumpy policy is non-monotone). Grows the singleton choice axis `1 → N_d`. |
| `Debit` | `WealthChangeStage` (`:liquid`) | `b ↦ max(b − outlay(d',d), ε)`. The down-payment + fixed cost, reading **both** `:durable_choice` (d') and `:durable` (old d). `outlay = p·(d'−d) + F·1{d'≠d}` (zero on keep). |
| `SetDurable` | `WealthChangeStage` (`:durable`) | Commits the choice `d ↦ d'`. |
| `Forget` | `ForgetfulSumStage` (`:durable_choice`) | Collapses the auxiliary axis. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` (`:liquid`) | Picks next-period liquid `b'`; `c = b_post − b'`. Flow utility is additively separable, `u_crra(c) + θ·log(d+1)`, with the durable service folded in via `utility_axes = (:durable,)`. |

Moments attached with `define_moments!`: `mean_liquid`, `mean_durable`. The
**adjustment rate** `∫ 1{d'≠d} dΛ` is a *transition* statistic — both adjusters
and keepers land on `cell.durable = d'`, so it is invisible to an `at_end`
integrand — and is recovered driver-side from the choice policy and the
distribution entering the choice (`steady_state.jl`).

## Why the auxiliary-choice axis

The down-payment `outlay(d', d)` reads **both** the chosen target `d'` and the
pre-choice stock `d`. A following `WealthChangeStage` placed after a plain gated
choice on `:durable` would see only the *post-choice* stock (`cell.durable = d'`),
not the old `d`, so it could not form the net stock change `d' − d` — the same
buyer-vs-continuing-owner distinguishability limitation documented in
`examples/durable_housing`. Routing the target through a separate
`:durable_choice` axis keeps both the old `d` (on `:durable`) and the chosen `d'`
(on `:durable_choice`) live until `Debit` has read them, then `Forget` collapses
the auxiliary axis. The fixed cost and down-payment ride the **following**
`Debit` WealthChange — never the choice reward — mirroring
`BuyHomeStage ∘ WealthChangeStage` and the two-asset-HANK deposit.

## A note on feasibility

The `:brute` `ArgmaxStage` requires every cell to have at least one feasible
action. A corner like cash-on-hand `= 0` with no durable to sell would otherwise
have none (keep gives `c = 0`, buying is unaffordable, nothing to sell). The
subsistence floor `ε` on post-transaction liquid (exactly the two-asset-HANK
device) keeps consumption feasible for every action; the consumption hit an
`ε`-floored over-purchase takes is what deters unaffordable adjustments, so no
explicit affordability gate is needed.

## Equilibrium

Returns and the wage are **exogenous** (partial equilibrium): there is no market
to clear, so the outer loop is a single inner V/Λ fixed-point solve
(`solve_steady_state_given_env!`). Impatience (`β(1+r_b) < 1`) plus the liquid
floor deliver a stationary distribution; the depreciation shock keeps the
adjustment margin live.

## Parameters and expected output

`N_b = 100` liquid grid on `[0, 12]`; 2-state income; durable sizes `[0,1,2,3]`;
`β = 0.93`, `σ = 2`, `θ = 0.55`, `p = 1.0`, `F = 0.12`, `π_dep = 0.15`. At the
baseline calibration:

```
mean liquid     ≈ 0.60
mean durable    ≈ 2.19
adjustment rate ≈ 0.118
```

An adjustment rate near 12% — infrequent, lumpy replacement — with the durable
stock spread across the interior levels rather than absorbed at the top, the
hallmark of an (S,s) inaction band.

## How to run

From the `HouseholdStages` directory:

```julia
julia --project=. examples/durable_liquid/steady_state.jl
```
