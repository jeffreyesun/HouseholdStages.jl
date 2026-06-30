# Cocco–Gomes–Maenhout (2005) — life-cycle portfolio choice

A finite-horizon household that **saves and chooses a risky portfolio share**
in the presence of non-tradable, hump-shaped labor income. Labor income is a
bond-like implicit asset (human wealth): when young, human wealth dwarfs
financial wealth, so the household tilts its *financial* portfolio heavily
toward equities; as financial wealth accumulates and human wealth runs down
with age, the risky share descends toward the Merton interior level. The
age-declining risky share is the CGM signature.

## The household block (existing stages only)

The within-period chain is the `examples/portfolio` chain, wrapped in the
`examples/life_cycle` finite-horizon `replicate_age` driver:

```
replicate_age(IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Portfolio, N; axis = :age)
```

| stage | library object | role |
|---|---|---|
| `IncomeShock` | `MarkovStage(:income)` | persistent earnings risk |
| `Receipt` | `WealthChangeStage` | cash-on-hand `x = b + y(age)·ε` (no `(1+r)` — see below) |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | picks invested wealth `b'`, `c = x − b'` |
| `Portfolio` | `MeanVarianceStage` | picks risky share `θ`; next wealth `b'·(R_f + θ·(R_k − R_f))` |

**No bespoke stage.** This is the four-stage portfolio chain under the
life-cycle product combinator. The one subtlety versus the plain `life_cycle`
example: `Receipt` carries **no** `(1+r)` factor, because the gross return is
delivered entirely by the portfolio stage (the risk-free leg `R_f` plays the
role of `1+r`). Baking a return into `Receipt` too would double-count it.

The finite-horizon solve — backward sweep threading the continuation value
across ages, then a forward cohort simulation — is example-side driver logic
in `steady_state.jl`, identical in shape to `life_cycle/steady_state.jl`.

## Reading the risky share per age

Each per-age component is the four-stage chain, so it carries **two**
policy-bearing leaves (savings and the risky share). `policy(::ChainStage)`
errors on more than one leaf, so the driver reads the risky-share policy
directly off the portfolio leaf — the last stage of each age component,
`comp[a].buffer.stages[end]`. The policy θ*[cell] is indexed by *post-savings*
wealth `b'`, so the age profile weights it by the distribution pushed forward
through the leading sub-stages (shock → receipt → savings), not the
start-of-period distribution.

## Results (`σ = 4`, premium ≈ 5%, N = 25)

```
age :  θ*(age)   mean financial wealth
  1 :  1.000     1.49
  4 :  1.000     0.85
  8 :  0.935     0.64     ← initial buffer bottoms out
 14 :  0.996     1.33
 19 :  0.687     1.98
 21 :  0.519     2.51     ← peak financial wealth
 24 :  0.489     1.09
```

- **High when young, declining with age.** θ* is at the cap (full equity) for
  the young, then descends monotonically over the back half of working life
  toward the **Merton interior share ≈ 0.28** as human wealth depletes. This
  is the CGM signature.
- Mean cross-sectional wealth ≈ 1.31; the financial-wealth profile is
  hump-shaped, peaking (~2.5) just before retirement.
- The terminal age reads θ* = 0 mechanically (no bequest ⇒ `b' = 0`).

## Limitation surfaced by the dogfood

The package's `MeanVarianceStage` allocates over **financial** wealth; it does
not augment human capital as an explicit riskless asset the way CGM's closed
form does. Consequently a household at (or near) zero financial wealth sits in
the extremely concave region of the CRRA value function and **de-risks** —
the opposite of the CGM young, who hold ~100% equity precisely because their
large bond-like human wealth substitutes for the riskless asset. To recover
the high-young share without a bespoke human-wealth-augmented portfolio stage
(forbidden here), newborns are seated with a financial endowment (≈ 1.5× peak
earnings, `w0_init`), which keeps them above the concavity floor through the
early ages. Fully reproducing CGM's flat-high young share with literally zero
initial wealth would require that bespoke stage.

## Run

```
julia --startup-file=no --project=. examples/cocco_gomes_maenhout/steady_state.jl
```
