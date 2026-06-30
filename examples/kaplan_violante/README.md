# Kaplan–Violante (2014) — wealthy hand-to-mouth

Two assets: a low-return **liquid** asset, freely adjusted, and a high-return
**illiquid** asset that can be rebalanced only by paying a **fixed (lumpy)
transaction cost** κ_f. The defining mechanism: many households optimally do NOT
pay the cost — they hold illiquid wealth but act hand-to-mouth in liquid,
generating large MPCs (the *wealthy hand-to-mouth*). Catalog §0 ◐ / §5(ii).

## Household block (existing stages only)

The same auxiliary-choice-axis pattern as `examples/two_asset_hank`, with the
adjustment cost made **fixed** (a non-convexity) and an explicit **no-adjust**
action added to the choice axis:

```
IncomeShock ∘ Receipt(:liquid) ∘ ChooseA' ∘ DebitLiquid ∘ CreditIlliquid
  ∘ Forget ∘ ConsumptionSavings(:liquid)
```

The `:illiquid_choice` axis has `N_a + 1` actions: indices `1..N_a` adjust to
`agrid[i]` and pay the fixed cost κ_f; index `N_a+1` is **no-adjust** (illiquid
accrues to `(1+r_a)a`, liquid untouched). `DebitLiquid` and `CreditIlliquid` are
`WealthChangeStage`s whose closures branch on the action; `Forget` is a
`ForgetfulSumStage`.

## Key finding

The fixed cost is a **non-convexity**, but the illiquid choice is resolved by a
**brute `ArgmaxStage`** over the discrete choice axis — which needs no convexity.
So the lumpy KV cost is just a different closure on the existing two-asset
pattern; no new primitive. The no-adjust option (the wealthy-HtM channel) is one
extra index on the same choice axis.

## Run

```
julia --startup-file=no --project=. examples/kaplan_violante/steady_state.jl
```

Solves: `mass(Λ) = 1.0`, illiquid share ≈ 0.96, **wealthy-HtM share ≈ 0.77**
(near-zero liquid, positive illiquid), ~249 VFI iters.
