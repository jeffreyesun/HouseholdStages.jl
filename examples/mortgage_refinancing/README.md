# Mortgage refinancing / cash-out, LTV-gated (fixed cost, (S,s))

A heterogeneous-agent homeowner with a fixed-value house `H` (subject to a
house-price shock) and a **mortgage balance** `m` (a few discrete levels)
alongside liquid wealth `a`. Each period the household may **refinance** —
adjust the balance up (cash-out, the borrowing margin) or down (prepay) — paying
a **fixed refinancing cost** `κ`, **gated by the loan-to-value constraint**
`m'/H ≤ θ_ltv`. The point of this example: the within-period problem is a
composition of **existing library stages**, no bespoke household stage.

## Household block

The within-period problem, in time order:

```
IncomeShock ∘ HousePriceShock ∘ Receipt ∘ [ ChooseM' ∘ LTVGate ∘ Pay ∘ SetMortgage ∘ Forget ] ∘ ConsumptionSavings
```

| Stage | Library stage | What it does |
|---|---|---|
| `IncomeShock` | `MarkovStage` (`:income`) | 2-state labour-income Markov. |
| `HousePriceShock` | `MarkovStage` (`:hp`) | Boom/bust house value. Shifts the LTV gate: in a bust, high balances are underwater and locked out of refinancing. |
| `Receipt` | `WealthChangeStage` (`:wealth`) | Cash on hand `a ↦ (1+r_a)·a + w·y`. |
| `ChooseM'` | `ArgmaxStage` (`:refi_choice`) | Picks the new balance `m'` onto a separate auxiliary axis (reward `0`; `:brute`, lumpy/non-monotone). Grows the singleton axis `1 → K`. |
| `LTVGate` | `BorrowingConstraintStage` | A cash-out (`m' > m`) is infeasible (`-Inf`) when `m'/hp > θ_ltv`. Keep and prepay are never gated, so the brute argmax always has a feasible action. |
| `Pay` | `WealthChangeStage` (`:wealth`) | `a ↦ max(a + (m'−m) − κ·1{m'≠m} − r_m·m', ε)`: principal change (cash-out +, prepay −), fixed cost, and interest on the new balance. Reads **both** `:refi_choice` (m') and `:mortgage` (old m). |
| `SetMortgage` | `WealthChangeStage` (`:mortgage`) | Commits the choice `m ↦ m'`. |
| `Forget` | `ForgetfulSumStage` (`:refi_choice`) | Collapses the auxiliary axis. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` (`:wealth`) | Picks next-period liquid `a'`; `c = a_post − a'`, CRRA. |

Moments attached with `define_moments!`: `mean_wealth`, `mean_mortgage`. The
**refinancing rate** `∫ 1{m'≠m} dΛ` is a *transition* statistic — both
refinancers and keepers land on `cell.mortgage = m'`, invisible to an `at_end`
integrand — recovered driver-side from the choice policy and the distribution
entering the choice (`steady_state.jl`).

## Why the auxiliary-choice axis (and not the default/buy-home pattern)

The recommended starting point was the simpler `DefaultStage`/`BuyHomeStage`
pattern: a gated choice on `:mortgage` followed by a `WealthChangeStage` reading
`cell.mortgage`. That pattern **cannot** express this model, for the same reason
`durable_housing` cannot charge a one-time stock price: a following stage sees
only the **post-choice** balance `m'`, not the old `m`, so it can compute neither
the principal cash flow `(m' − m)` nor the one-time fixed cost `κ·1{m'≠m}` (both
need to compare old and new). The default example sidesteps this because its
consequences (debt reset, income haircut, exclusion spell) are functions of the
**new** status alone, and its `−χ` is a *per-period* flow cost, not a one-time
transition cost.

Routing the new balance through a separate `:refi_choice` axis keeps both the
old `m` (on `:mortgage`) and the chosen `m'` (on `:refi_choice`) live until `Pay`
has read them, so the principal delta and the one-time fixed cost are both
expressible. This is the same Route-A resolution used by `two_asset_hank`,
`habit`, and `durable_liquid`. The fixed cost rides the **following** `Pay`
WealthChange — never the choice reward.

## Mechanism

`r_m > r_a` makes debt costly, so the household wants to prepay when liquid-rich;
the no-unsecured-borrowing constraint (`a' ≥ 0`) makes a cash-out refinance the
only borrowing margin, so income risk drives cash-out when liquid-poor. The
fixed cost `κ` makes adjustment **infrequent and lumpy** — an (S,s) band in the
mortgage balance. The LTV cap binds in the bust state, where high balances are
underwater and cannot cash out. Relaxing the cap (`θ_ltv` large) is a live
margin: mean mortgage rises `0.83 → 0.92` and the refi rate `0.088 → 0.106`,
confirming the gate is non-vacuous.

## Equilibrium

Prices (`r_a`, `r_m`, `w`) are **exogenous** (partial equilibrium): no market to
clear, so the outer loop is a single inner V/Λ fixed-point solve
(`solve_steady_state_given_env!`).

## Parameters and expected output

`N_a = 100` liquid grid on `[0, 10]`; 2-state income; 2-state house price
`[3, 5]`; mortgage levels `[0, 1, 2]`; `β = 0.94`, `σ = 2`, `r_a = 0.02`,
`r_m = 0.05`, `κ = 0.05`, `θ_ltv = 0.50`. At the baseline calibration:

```
mean wealth      ≈ 0.53
mean mortgage    ≈ 0.83
refinancing rate ≈ 0.088
```

A refinancing rate near 9% — infrequent, lumpy adjustment — with the balance
distribution spread across the interior levels.

## How to run

From the `HouseholdStages` directory:

```julia
julia --project=. examples/mortgage_refinancing/steady_state.jl
```
