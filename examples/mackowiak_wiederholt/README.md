# Maćkowiak–Wiederholt (2009, 2015) — single-margin attention

A savings household that allocates **finite attention to its idiosyncratic state** while an
**aggregate condition** evolves exogenously — the single-margin reading of Maćkowiak & Wiederholt
(2009, AER, "Optimal Sticky Prices under Rational Inattention"; 2015, REStud, "Business Cycle
Dynamics under Rational Inattention"). The within-period problem is **five existing `HouseholdStages`
stages, no bespoke household stage** (Part 3: literature household blocks expressible from the library
alone).

## Household block

In time order:

```
AggShock ∘ IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Attention
```

| stage | library stage | does |
|---|---|---|
| `AggShock` | `MarkovStage` | evolve the aggregate productivity state on `:z` |
| `IncomeShock` | `MarkovStage` | draw next idiosyncratic income on `:income` |
| `Receipt` | `WealthChangeStage` | `b ↦ (1+r)·b + w·z·y` (aggregate `z` scales the wage) |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | pick next wealth `b'`, `c = b_in − b'`, CRRA |
| `Attention` | `ScaleVarianceStage` | pick dispersion `θ` of `b' ↦ b' + θ·ξ` (ξ mean-zero) at cost `c(θ) = λ·θ²` |

The agent's one attention margin is precision about its own carried wealth state: `θ = 0` is perfect
attention (no noise, no cost), larger `θ` is a noisier, cheaper read. The cost `c(θ) = λ·θ²` is a
**stand-in for `λ·KL(θ)`** — the information-cost placeholder, not a literal Shannon term.

## What is faithful, and what is the ◐/G3 gap

The genuine Maćkowiak–Wiederholt mechanism is the **allocation of a finite attention capacity across
multiple signals** — aggregate vs idiosyncratic — coupling their precisions through **one** budget
constraint. That **coupled multi-signal attention budget is a recorded ◐/G3-adjacent gap**
(MODEL_CATALOG §2; gaps item 5): a single Shannon constraint over several precisions is **not one
univariate streaming stage**. Each per-axis `ScaleVarianceStage` optimises its own `θ`
independently — there is no shared-budget coupler in the per-axis streaming vocabulary that ties the
precision spent on `z` to the precision spent on the idiosyncratic state. Expressing it would need a
coupled-constraint generalisation the library does not offer.

What **is** faithful, and what this example builds, is the **single-margin household**: one attention
choice (here, to the idiosyncratic wealth state), priced at `λ·θ²`, with the aggregate condition `z`
entering exogenously through an ordinary `MarkovStage`. This is the ✅ per-axis member of the MW
entry; the coupled budget is the ◐ part. The attention gradient below is the genuine MW comparative
static — just realised on one margin rather than split across a joint budget.

## Why θ\* is interior (the economics)

Identical to `examples/rational_inattention`. A mean-preserving spread on a globally concave
continuation picks `θ = 0` (Jensen, plus a positive cost). The bite is the **borrowing constraint**:
for poor, near-constrained households the continuation in wealth is locally **convex** (a low draw is
floored at the constraint, a high draw escapes it), so a small dispersion carries option value; the
cost `λ·θ²` disciplines how much is taken.

## What the run shows

`julia --project=. examples/mackowiak_wiederholt/steady_state.jl`:

- **Attention gradient in wealth.** `θ*` = 1.0 (maximal dispersion / least precise attention) for the
  poorest decile, `θ*` = 0.0 (perfect attention) for the richest decile — the constrained poor hold
  the convex region where dispersion has option value.
- **The RI comparative static.** Mean chosen dispersion falls monotonically as the information cost
  `λ` rises (`θ̄* ≈ 0.32 → 0.31 → 0.26 → 0.17` for `λ = 0, 0.001, 0.01, 0.05`); the share of cells
  choosing positive dispersion shrinks with `λ`.

Returns are exogenous (partial equilibrium): no market clears, so the outer loop is a single
`solve_steady_state_given_env!`. The borrowing constraint plus impatience (`β·(1+r) < 1`) deliver a
stationary distribution.

## Literature

Maćkowiak & Wiederholt (2009, AER); Maćkowiak & Wiederholt (2015, REStud); Sims (2003, JME),
"Implications of Rational Inattention." The continuous-precision (KL/`ScaleVarianceStage`) sibling of
the discrete-posterior (entropy/`LogitChoiceStage`) RI in `examples/discrete_ri`.
