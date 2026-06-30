# Two-asset HANK (Kaplan–Moll–Violante 2018)

Liquid asset `b` (low return `r_b`, free to adjust) and illiquid asset `a` (high return `r_a`, costly
to adjust). The illiquid choice `a'` must set the illiquid axis **and** debit liquid by the deposit
`d = a' − (1+r_a)a` (+ convex cost `χ`). That cross-axis flow is what halted naive attempts; it **is**
expressible from existing stages via the **auxiliary-choice-axis pattern** (cf. `examples/habit`).

## Household block (existing stages only, no bespoke stage)

```
IncomeShock ∘ Receipt ∘ [ ChooseA' ∘ DebitLiquid ∘ CreditIlliquid ∘ Forget ] ∘ ConsumptionSavings
```

| stage | library stage | role |
|---|---|---|
| `ChooseA'` | `ArgmaxStage(:illiquid_choice)` | pick `a'` onto a separate axis |
| `DebitLiquid` | `WealthChangeStage(:liquid)` | `b ↦ b − d − χ(d)` — reads **both** `:illiquid_choice` (a') and `:illiquid` (old a) |
| `CreditIlliquid` | `WealthChangeStage(:illiquid)` | commit `a ↦ a'` |
| `Forget` | `ForgetfulSumStage(:illiquid_choice)` | collapse the auxiliary axis |
| `ConsumptionSavings` | `ConsumptionSavingsStage(:liquid)` | consume/save from post-deposit liquid (β here) |

Routing `a'` through the separate `:illiquid_choice` axis keeps the old illiquid `a` and the chosen
`a'` both live, so `DebitLiquid` can form the net deposit `d`. A subsistence floor `ε` on post-deposit
liquid keeps consumption feasible.

## Verified
`test/test_example_two_asset_hank.jl` pins the illiquid block's deposit (the cross-axis flow) to its
brute reference `V(b,a)=max_{a'} V_next(b−d−χ, a')` **to machine precision**, and solves the full
model to a stationary steady state.

## Caveats
- The `:illiquid_choice` grid makes the intermediate tensor `N_a×` larger (gridded). A future "one
  choice, two axes" primitive — or the portfolio-value reformulation (state `(W, a)`, liquid `= W−a`;
  see `MODEL_CATALOG.md`) — would avoid it.
- This is a quadratic-cost version (smooth adjustment); a fixed cost would give the lumpy (S,s) margin.

## Run
```julia
julia --project=. examples/two_asset_hank/steady_state.jl
```
