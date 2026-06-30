# Vintage technology adoption / replacement — regenerative stopping

A heterogeneous-agent household that operates a productive durable / technology
of **vintage** `v` whose quality decays stochastically each period
(obsolescence). Every period the agent chooses **keep** (let the vintage drift
down) or **adopt** (pay a fixed cost `F` and reset the vintage to the newest
level). Adopting pays off only once the vintage has decayed far enough that the
output gain net of the cost turns positive — the §5(i) **regenerative optimal
stopping** structure of **Rust (1987)** bus-engine replacement and
Cooper–Haltiwanger capital replacement, embedded here in an income-fluctuations
consumption-savings household.

As with every §5 example, the entire within-period block is a `∘`-composition of
**existing library stages** — no bespoke stage.

## The coupling, and why the auxiliary-choice-axis pattern is needed

The adopt choice must move **two** axes at once: it resets the `:vintage` axis to
the top *and* charges a resource cost `F` on the `:wealth` axis — and the cost
must hit a **fresh adopter only**, never a continuing top-vintage keeper sitting
on the identical post-choice cell. A plain gated `ArgmaxStage` on `:vintage`
cannot express this: a fresh adopter (old vintage → top) and a standing
top-vintage owner land on the same `vintage = v_top` cell, so any following
`WealthChangeStage` charges them identically — the exact distinguishability wall
documented in `durable_housing`'s one-time-price note.

The fix is the **auxiliary-choice-axis pattern** (Route A, as in
`examples/habit`): route the keep/adopt decision through a transient
`:adopt_choice` axis, so downstream stages read the *decision* — not just the
post-choice vintage — and can reset the vintage **and** charge the cost to
adopters only, with the pre-choice vintage still live.

## Household block

The within-period problem, in time order:

```
IncomeShock ∘ Choose ∘ SetVintage ∘ Receipt ∘ PayCost ∘ Forget ∘ Savings ∘ Depreciate
```

| Stage | Library stage | What it does |
|---|---|---|
| `IncomeShock` | `MarkovStage` (`:income`) | 2-state labour-income risk. |
| `Choose` | `ArgmaxStage` (`:adopt_choice`) | Grows the choice axis 1→2 ({keep, adopt}); reward 0, so it just compares the two continuations `V[keep]`, `V[adopt]`. |
| `SetVintage` | `WealthChangeStage` (`:vintage`) | Adopters reset to `v_top`; keepers stay (reads BOTH the choice and the old vintage). |
| `Receipt` | `WealthChangeStage` (`:wealth`) | Cash-on-hand `(1+r)·b + w·y + θ·v` on the **post-adoption** vintage. |
| `PayCost` | `WealthChangeStage` (`:wealth`) | Adopters pay `F` out of cash-on-hand (after receipt, so `c > 0` stays feasible; reads the choice → adopters only). |
| `Forget` | `ForgetfulSumStage` (`:adopt_choice`) | Collapse the auxiliary choice axis 2→1. |
| `Savings` | `ConsumptionSavingsStage` (`:wealth`) | Pick `b'`, `c = cash − b'`, CRRA over `c`. |
| `Depreciate` | `MarkovStage` (`:vintage`) | Obsolescence drift — the vintage falls one level w.p. `π_dep`, carried into next period. |

Structurally this is the same kernel-choice sandwich as `habit`
(`Choose ∘ {axis transforms reading the choice} ∘ ForgetfulSum`), applied to a
stopping problem rather than a smooth habit.

Moments attached (`define_moments!`): `mean_vintage = ∫ v dΛ` and `top_share =
∫ 1{v = v_top} dΛ`. The per-period **adoption rate** is not a moment of the
end-of-period distribution (the `:adopt_choice` axis is collapsed by `Forget`),
so the driver reads it off the `Choose` policy weighted by the distribution
*entering* the choice.

## The (S,s)-style replacement policy

An agent already at the top vintage never adopts (paying `F` to go top→top is
strictly dominated), so adoption is *selective*: agents replace only once the
vintage has decayed enough that the discounted output gain exceeds `F`. The
decision also depends on wealth (CRRA marginal utility), giving rich/poor
heterogeneity in the replacement threshold.

## Equilibrium

`r, w` and the technology parameters `θ, F` are **exogenous** (partial
equilibrium): no market to clear, so the "outer loop" is a single inner V/Λ
fixed-point solve (`solve_steady_state_given_env!`). The obsolescence Markov
keeps the lower vintages populated so the adopt choice stays live.

## Parameters and expected output

`β = 0.93`, `σ = 2`, `r = 0.03`, `w = 1`, vintage output scale `θ = 1`, fixed
cost `F = 1.0`, vintage grid `[1.0, 1.4, 1.8, 2.2]`, obsolescence `π_dep = 0.25`,
2-state income, `N_w = 120` log wealth grid on `[0, 30]`. At baseline:

```
mean vintage  ≈ 2.09   (grid top 2.2)
top-vint. sh  ≈ 0.73
adoption rate ≈ 0.24   per period
```

Most mass sits near the newest vintage (frequent replacement), with ~24% of
agents replacing each period — essentially all non-top agents adopt, the top
agents never do, a clean regenerative-stopping churn.

## How to run

```julia
julia --project=. examples/vintage_replacement/steady_state.jl
```
