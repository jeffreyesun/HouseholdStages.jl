# Taste Shocks — Preference Risk in a Bewley Economy

A standard incomplete-markets self-insurance economy with an **exogenous
preference (taste) shock** layered onto the income process. Beyond income risk,
the household's flow utility is shifted each period by a persistent taste state
(a "marginal value of being here" shock). The taste shock is a pure additive
`UtilityStage` shifter on the value function, driven by its own Markov axis —
built entirely from existing library stages.

## Household block — existing library stages only

```
TasteShock ∘ IncomeShock ∘ IncomeReceipt ∘ UtilityStage(taste) ∘ ConsumptionSavingsStage
```

| Stage (model role) | Library stage | What it does |
|---|---|---|
| `TasteShock` | `MarkovStage` (axis `:taste`) | Preference Markov draw `P_taste`. |
| `IncomeShock` | `MarkovStage` (axis `:income`) | Idiosyncratic income draw `P_y`. |
| `IncomeReceipt` | `IncomeStage` | `a ↦ (1+r)a + w·y`. |
| `UtilityStage(taste)` | `UtilityStage` | Adds the taste state's flow value (reads the `:taste` axis); an additive shift to V. |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` | Choose next wealth; budget `c = a_in − a_end`, CRRA. |

The taste enters as an **additive** flow-value shifter (the value of being in the
taste state). A multiplicative twist on consumption felicity would instead pass
`utility_axes = (:taste,)` into `ConsumptionSavingsStage`; the additive form is
the clean composable demonstration of an exogenous taste process moving V.

## Outer loop

Fixed-`r` partial equilibrium (the Bewley framing): a **single** inner V/Λ solve
at the exogenous return `r < 1/β − 1` delivers the stationary **joint**
distribution over `(wealth, income, taste)`. No market clearing.

## Result (default calibration)

```
β = 0.95, σ = 2.0, r = 0.03, taste grid [-0.10, 0.0, 0.10]
ΣΛ = 1.000000,  A_mean (buffer stock) = 2.49,  VFI iters = 317
```

The non-degenerate ergodic distribution confirms the buffer-stock mechanism
survives the added preference risk.

## Run

```julia
# from HouseholdStages/
julia --project=. examples/taste_shocks/steady_state.jl
```
