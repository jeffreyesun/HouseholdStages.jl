# Constantinides (1990) internal habit formation

Felicity is CRRA over the consumption **surplus** `c − γ·S`, where `S` is an
**internal** habit stock built from the household's own past consumption
(Constantinides 1990, *"Habit Formation: A Resolution of the Equity Premium
Puzzle"*). The same chosen `c = x − b'` both is consumed now **and** builds next
period's stock `S' = (1−δ_S)·S + c`. This is the exact machinery of
`examples/habit` — the **auxiliary-choice-axis pattern** routing the single
savings choice so it can set liquid wealth *and* the habit stock — with a
one-function felicity change: Becker–Murphy adjacent complementarity (`√c + α·c·S`)
becomes the Constantinides internal-habit surplus `u(c − γS)`.

## Household block (existing stages only, no bespoke stage)

```
IncomeShock ∘ Receipt ∘ Choose ∘ Utility ∘ Discount ∘ HabitUpdate ∘ SetLiquid ∘ Forget
```

| stage | library stage | role |
|---|---|---|
| `Choose` | `ArgmaxStage(:savings_choice)` | pick `b'` **onto a separate axis** (the argmax populates it) |
| `Utility` | `UtilityStage` | felicity `u(c − γS)`, `c = x − b'` (reads `:wealth`, `:savings_choice`, `:habit`) |
| `Discount` | `TimeDiscountingStage` | β on the continuation only |
| `HabitUpdate` | `WealthChangeStage(:habit)` | `S' = (1−δ_S)S + (x − b')` — reads the **old** wealth `x` *and* the choice |
| `SetLiquid` | `WealthChangeStage(:wealth)` | commit `wealth ← b'` |
| `Forget` | `ForgetfulSumStage(:savings_choice)` | collapse the auxiliary axis |

The wiring is byte-for-byte the `habit` chain; **only the felicity differs**.
Routing `b'` through a separate `:savings_choice` axis keeps the pre-choice
wealth `x` and the chosen `b'` both live, so `HabitUpdate` can read both and form
`c`. Pre-choice stages see `:savings_choice` as a singleton; the choice block
grows it `1→N_w` and `Forget` collapses it back.

## The one departure from strict `−∞` subsistence: a consumption floor

Constantinides subsistence wants felicity `= −∞` when `c − γS ≤ 0`. That cannot
go through this pattern as-is, and the reason is precise and worth recording:

- The auxiliary pattern grows the savings axis through a (brute)
  **`ArgmaxStage`** (`Choose`), and an origin cell with **no finite-reward
  action carries value `−∞`** (`argmax.jl`: the `typemin` masking convention).
- The **minimum-wealth cell** structurally has `c = 0`: the cash-on-hand floor
  and the savings floor are the *same* grid point, so the most the agent can
  consume there is `x − b'_min = 0`. Under strict subsistence that cell's entire
  column is `−∞` (`surplus = −γS ≤ 0`), poisoning the value at the grid bottom.
- `examples/habit` is immune **only** because Becker–Murphy `√c + α·c·S` is
  **finite at `c = 0`** (it returns `0`). A CRRA surplus is `−∞` there.

The resolution is the standard quantitative-subsistence device: **floor the
surplus at a small `c_floor > 0`**, `u = u_crra(max(c − γS, c_floor))`. Felicity
stays finite (the `−∞` region becomes a steep-but-finite penalty), so `Choose`'s
assertion is satisfied, while the floored region carries no equilibrium mass.
The driver verifies this: at the baseline the mass at or below the floor is
`≈ 2.5×10⁻⁵`, i.e. the strict-`−∞` region is effectively unreachable, so the
solved model is numerically the Constantinides subsistence model.

## Equilibrium

Partial equilibrium (exogenous `r`): a single inner V/Λ solve, no market to
clear. The subsistence reference generates a strong precautionary motive — the
household saves to keep the surplus comfortably positive after a bad income
shock, the consumption-smoothing channel that, in asset-pricing form, raises the
price of risk.

## Parameters and expected output

`β = 0.94`, `σ = 2`, `γ = 0.2`, `δ_S = 0.6`, `r = 0.03`, 2-state income
`{0.85, 1.15}`, `N_w = N_S = 24`, `w_max = 14`, `S_max = 40`,
`c_floor = 10⁻³`. At baseline:

```
mean wealth = 4.03
mean habit  = 1.87
min surplus : floored region carries mass ≈ 2.5e-5 (unreachable)
```

## How to run

```julia
julia --project=. examples/constantinides_habit/steady_state.jl
```
