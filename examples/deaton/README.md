# Deaton (1991) — buffer-stock saving

Deaton's (1991) buffer-stock saving model. An **impatient** household
(`β(1+r) < 1`) faces a persistent AR(1) labour-income process and a hard
borrowing constraint (`a ≥ 0`). Impatience makes it want to borrow against
future income; the constraint forbids it; the precautionary motive then
makes it hold a small buffer stock of assets, drawn down in bad income
spells and rebuilt in good ones. Assets stay low and the constraint binds
frequently — the signature buffer-stock behaviour.

## The household block

The within-period problem is the canonical three-stage spine:

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

| Chain role | Library stage | What it does |
|---|---|---|
| `IncomeShock` | `MarkovStage` (`axis = :income`) | AR(1) income draw, discretized **offline** by Rouwenhorst (grid + matrix pre-computation, not a stage). |
| `IncomeReceipt` | `IncomeStage` (`axis = :wealth`) | Receipt `a ↦ (1+r) a + w·y`. |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` (`axis = :wealth`) | Choose next-period assets; the grid floor `a_min = 0` **is** the hard borrowing constraint; CRRA utility. |

No bespoke stage — three existing exported stages composed with `∘`. The
AR(1) is discretized by a small `rouwenhorst(N, ρ, σ_ε)` helper in
`model.jl` (ordinary parameter pre-computation: it returns a grid and a
row-stochastic matrix, which are then handed to `MarkovStage`). Two
moments are attached: `A_mean` (the buffer stock) and `frac_constrained`
(mass at the binding `a = 0` constraint).

## Equilibrium framing

The return `r` is **fixed and exogenous** with `β(1+r) < 1` (impatience),
so a single `solve_steady_state_given_env!` delivers the stationary
buffer-stock distribution — no market is cleared.

## Result

Default `DeatonParams`: β = 0.95, σ = 2.0, **r = 0.03 fixed**
(`β(1+r) = 0.9785 < 1`); AR(1) with ρ = 0.90, σ_ε = 0.10 on 7 Rouwenhorst
states; `N_a = 250` log-spaced asset points on `[0, 40]`.

```
r                  = 0.0300   β(1+r) = 0.9785  (< 1 ⇒ impatient)
ΣΛ                 = 1.000000
A_mean (buffer)    = 0.5940
frac at constraint = 0.2312
```

The small buffer (`A_mean ≈ 0.59`, well under one period's mean income)
together with ~23% of households at the constraint is the textbook
buffer-stock signature: impatience keeps assets low, and the precautionary
motive alone holds the buffer up.

## Run

```bash
julia --project=. examples/deaton/steady_state.jl
```

A single fixed-`r` solve; about 7 seconds after first compilation.

## Files

- `model.jl` — parameters, the `rouwenhorst` helper, layout, the three-
  stage chain, CRRA utility, `deaton_env`, and the buffer-stock moments.
- `steady_state.jl` — the single fixed-`r` solve driver and its report.
