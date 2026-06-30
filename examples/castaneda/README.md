# Castañeda–Díaz-Giménez–Ríos-Rull (2003) — earnings + OLG

A persistent idiosyncratic **earnings** Markov, an OLG age structure with
**retirement** and **stochastic death**, and **newborns who inherit the
accidental bequests** of the deceased.

## Household block — clean composition of existing stages

```
MarkovStage(:earnings) ∘ MarkovStage(:age, sub-stochastic)
                       ∘ IncomeStage ∘ ConsumptionSavingsStage
```

over a `(wealth, earnings, age)` layout. Every element is an existing exported
stage, parameterized only by matrices and closures:

- `MarkovStage(:earnings)` — persistent earnings draw (matrix `P_e`).
- `MarkovStage(:age)` with a **sub-stochastic** survival-hazard matrix —
  deterministic aging conditional on survival, a per-period death hazard in
  retirement, and certain death at the max age (all-zero last row). Rows that
  sum to `< 1` bleed the deceased off the age axis. (`AdvanceAgeStage` is
  exactly this primitive with a shift matrix; handing `MarkovStage(:age)` the
  hazard matrix directly makes death genuinely stochastic.)
- `IncomeStage` — `(1+r)·a + y`, with `y = pension` for retirees
  (`age ≥ retire_age`, read off the `age` coordinate in the receipt closure)
  and `w·earnings` for workers.
- `ConsumptionSavingsStage` — next-period wealth, CRRA felicity.

**The within-period block composes cleanly** — no bespoke stage, no kernel
reaching, no second-asset coupling.

## Custom demographics outer loop (expected, in `steady_state.jl`)

The block's age Markov *deletes* the deceased; the driver closes the
demographics the block deliberately does not. Per forward pass:

```
Λ⁺ = block.forward(Λ);   Λ⁺[:, :, age=1] += beq(Λ) ⊗ π_earn
```

`beq(Λ)[w]` is the wealth marginal of this pass's deceased (retirees failing
the survival draw + the entire max-age cohort). That mass is re-injected as
newborns at age 1, carrying the deceased's wealth as accidental bequests, with
earnings drawn from the ergodic earnings distribution. Injected mass ≡ deceased
mass, so `ΣΛ = 1` exactly. This cohort loop is rolled by hand, mirroring the
life_cycle finite-horizon sweep.

## What is full vs simplified

**Full:** earnings Markov; OLG aging; stochastic death (survival hazard +
certain death at max age); retirement with pension income; newborn inflow with
`ΣΛ = 1` renormalization; accidental bequests flowing to newborns (proportional
to the deceased wealth distribution).

**Simplified (vs. the JPE paper):** retirement timing is deterministic-by-age
rather than a stochastic retirement shock (death is the stochastic transition
here); an illustrative 3-state earnings process rather than the paper's
calibrated process matching U.S. earnings/wealth moments; bequest wealth uses
the deceased's start-of-period wealth marginal as a one-period proxy for
terminal assets; fixed-`r` partial equilibrium — no government tax/transfer
block and no calibration loop (the paper's central contribution).

## Solve

```
julia --startup-file=no --project=. examples/castaneda/steady_state.jl
```

Typical output: `ΣΛ = 1.00000000`, VFI ≈ 8 iters (one per age stage),
demographics ≈ 275 iters, a hump-shaped mean-wealth-by-age profile (rising
through working life, dissaving in retirement), and a retiree population share
thinned by the death hazard.
