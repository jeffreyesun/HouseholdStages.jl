# Habit formation / rational addiction (Becker–Murphy 1988)

Felicity depends on consumption relative to an **addiction stock** `S`, and the *same* chosen
consumption `c = x − b'` both is consumed now and builds next period's stock `S' = (1−δ_S)S + c`. So a
single savings choice must set **two** axes at once — liquid wealth *and* the habit stock. That
naively looks un-expressible (the deposit/flow `c` is gone once the choice overwrites wealth), and two
sub-agents halted on it. It **is** expressible from existing stages, via the **auxiliary-choice-axis
pattern**.

## Household block (existing stages only, no bespoke stage)

```
IncomeShock ∘ Receipt ∘ Choose ∘ Utility ∘ Discount ∘ HabitUpdate ∘ SetLiquid ∘ Forget
```

| stage | library stage | role |
|---|---|---|
| `Choose` | `ArgmaxStage(:savings_choice)` | pick `b'` **onto a separate axis** (the argmax populates it) |
| `Utility` | `UtilityStage` | felicity `u(c,S)`, `c = x − b'` (reads `:wealth`, `:savings_choice`, `:habit`) |
| `Discount` | `TimeDiscountingStage` | β on the continuation only |
| `HabitUpdate` | `WealthChangeStage(:habit)` | `S' = (1−δ_S)S + (x − b')` — reads the **old** wealth `x` *and* the choice |
| `SetLiquid` | `WealthChangeStage(:wealth)` | commit `wealth ← b'` |
| `Forget` | `ForgetfulSumStage(:savings_choice)` | collapse the auxiliary axis |

The trick: routing `b'` through a separate `:savings_choice` axis keeps the pre-choice wealth `x` and
the chosen `b'` **both live**, so `HabitUpdate` can read both and form `c`. Structurally this is the
kernel-choice sandwich (`Collapse ∘ transforms ∘ ForgetfulSum`). Pre-choice stages see
`:savings_choice` as a singleton; the choice block grows it `1→N_w` and `Forget` collapses it back.

## Verified

`test/test_example_habit.jl` pins the chain's backward value to the brute Bellman
`V(x,S) = max_{b'} u(x−b', S) + β·V(b', (1−δ_S)S + (x−b'))` **to machine precision**, and solves the
full income-fluctuation model to a stationary steady state.

## Caveats
- The `:savings_choice` grid makes the intermediate tensor `N_w×` larger (gridded). A future "one
  choice, two axes" primitive would collapse the memory.
- The savings policy can be non-monotone, so this uses the brute `ArgmaxStage`.

## Run
```julia
julia --project=. examples/habit/steady_state.jl
```
