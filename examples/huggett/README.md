# Huggett (1993) — Pure-Exchange Bond Economy

A one-period risk-free bond in **zero net supply**, an incomplete-markets
exchange economy. Households smooth a fluctuating endowment by trading a
single bond subject to a borrowing limit; the equilibrium interest rate
is the price at which aggregate bond demand clears to zero.

## Household block — existing library stages only

The within-period problem is the **same three-stage decomposition as
Aiyagari** (canonical L03/L04). No bespoke household stage is rolled here.

| Stage (model role) | Library stage | What it does |
|---|---|---|
| `IncomeShock` | `MarkovStage` | Resolves the endowment Markov draw on the `:income` axis (transition `P_y`). |
| `IncomeReceipt` | `WealthChangeStage` | Deterministic `a ↦ (1+r) a + y`: bonds pay gross return `1+r`, plus the pure endowment `y` (no wage/labor). |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` | Chooses next-period bond holdings on the `:wealth` grid; implicit budget `c = a_in − a_end`. |

Chain: `shock ∘ receipt ∘ savings`. The block is **identical** to
Aiyagari's; the only model-side differences are the receipt law (`+ y`
endowment rather than `+ w·y` wage income) and the attached moment
(`A_supplied = ∫ a dΛ`, aggregate bond demand).

## What makes it Huggett, not Aiyagari (the outer loop)

The distinction lives entirely in the driver (`steady_state.jl`), not the
household block:

- **Zero net supply.** No capital, no production. Equilibrium is the rate
  `r` at which aggregate bond demand `A_supplied(r) = ∫ a dΛ` clears to
  **zero** — Aiyagari instead clears `K` to a production capital demand.
- **Borrowing limit.** The asset grid spans negative wealth `[a_min, 0]`
  (the ad-hoc limit `ā = a_min`) glued to a log-spaced positive region.
- **`A_supplied(r)` is increasing in `r`**, so the driver brackets the
  root in `[r_lo, r_hi]` and **bisects**, reusing
  `solve_steady_state_given_env!` for the per-`r` inner V/Λ solve.

The asset grid is linear in the borrowing region (where the constraint
binds and policies are sharply nonlinear) and log-spaced above zero, so
the post-receipt point `(1+r)a + y` stays inside the grid for active
cells (the Aiyagari log-grid argument for `WealthChangeStage.backward`).

## Equilibrium

With the default calibration (β = 0.96, CRRA σ = 1.5, two-state
endowment, `a_min = −2`), the bond market clears at

```
r ≈ 0.0157   (autarky rate 1/β − 1 = 0.0417)
∫ a dΛ ≈ 0   (zero net supply)
```

The equilibrium rate sits **strictly below** the time-preference rate
`1/β − 1`: precautionary demand for the bond depresses its return, the
hallmark Huggett result.

## Run

```julia
# from HouseholdStages/
julia --project=. examples/huggett/steady_state.jl
```

prints the bisection trace, the cleared `r`, the residual bond demand,
and total mass `ΣΛ`. The regression test is
`test/test_example_huggett.jl`.
