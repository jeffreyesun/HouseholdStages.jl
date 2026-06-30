# Life-cycle / OLG consumption-savings — finite horizon

A finite-horizon life-cycle household à la **Gourinchas–Parker (2002)** and
**Cocco–Gomes–Maenhout (2005)**: an `N`-period household that saves against a
hump-shaped deterministic earnings profile and persistent income risk, with no
bequest. The household block is **one existing library object** —
`replicate_age(…)` of the Aiyagari within-period chain — with **no bespoke
household stage** (Part 3: literature household blocks expressible from the
library alone).

## Household block

The block is a `ProductStage` stacking `N` age-specific copies of the
within-period chain along an `:age` axis:

```
replicate_age( IncomeShock ∘ Receipt ∘ ConsumptionSavings , N; axis = :age )
```

| stage | library stage | does |
|---|---|---|
| `replicate_age(…, N; axis=:age)` | `ProductStage` | stack `N` age-slices on the `:age` axis |
| `IncomeShock` | `MarkovStage` | persistent earnings risk on the `:income` axis |
| `Receipt` | `WealthChangeStage` | `b ↦ (1+r)·b + y(age)·ε` (cash-on-hand; `y(age)` is the hump) |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | pick next wealth `b'`, `c = x − b'`, CRRA utility |

## Why a finite-horizon driver (and why it's not a bespoke stage)

A life-cycle model is **not** a stationary steady state — there is no
VFI-to-fixed-point. A `ProductStage`'s own `backward!` runs each age-slice
**independently** (a block-diagonal direct sum); it does not thread
age-`(a+1)`'s continuation value into age-`a`. The wiring that a finite horizon
needs is therefore rolled in `steady_state.jl` as plain example outer-loop logic
(explicitly allowed — same status as a tatonnement loop), driving the
ProductStage's per-age components `hh.buffer.components[a]`:

1. **Backward induction.** With no bequest (`V_{N+1} = 0`), sweep `a = N…1`,
   feeding age-`(a+1)`'s continuation value into age-`a`'s component. This seats
   each age's savings policy on its own buffer.
2. **Forward cohort simulation.** A unit cohort of newborns (zero wealth,
   ergodic income) is pushed `a = 1…N` through the per-age components; the
   age-`a` distribution is stored in slice `a` of a stacked `(N_w, n_ε, N)` Λ.

No new stage, kernel, or per-cell value/transition logic is introduced — only
existing stages' `backward!` / `forward!` verbs, called age-by-age.

## Equilibrium

**Partial equilibrium** (the standard GP/CGM setup): the real rate `r` and the
age-earnings profile `y(age)` are exogenous, so there is no market to clear. The
"solve" is the single backward + forward pass above.

## Run

```julia
julia --project=. examples/life_cycle/steady_state.jl
```

Reports per-age cohort mass (conserved at 1.0), cross-sectional mean wealth, and
the wealth profile. With the default calibration the profile is the textbook
**life-cycle hump**: zero wealth at birth, accumulation through working life,
a peak just after retirement, and decumulation thereafter.
