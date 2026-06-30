# Livshits–MacGee–Tertilt (2007) — life-cycle consumer bankruptcy

A finite-horizon life-cycle household à la **Livshits, MacGee & Tertilt
(2007)**, "Consumer Bankruptcy: A Fresh Start": an `N`-period household with
persistent earnings risk and a hump-shaped age-earnings profile that can borrow
(`a_min < 0`) and each period in good standing choose **repay vs file** for
Chapter-7 bankruptcy. Filing discharges all debt (assets reset to 0) at the cost
of an exclusion/garnishment spell. The household block is the **same six-stage
chain as `examples/default`**, wrapped in `replicate_age` and solved with the
`life_cycle` finite-horizon driver — **no bespoke household stage**.

## Household block

```
replicate_age( IncomeShock ∘ DefaultChoice ∘ DebtReset ∘ Receipt ∘
               ConsumptionSavings ∘ Readmission , N; axis = :age )
```

| stage | library stage | does |
|---|---|---|
| `replicate_age(…, N; axis=:age)` | `ProductStage` | stack `N` age-slices on the `:age` axis |
| `IncomeShock` | `MarkovStage` | persistent earnings risk on `:income` (hump `y(age)` rides `env`) |
| `DefaultChoice` | `DefaultStage` | gated repay/file argmax on the 2-level `:status` axis |
| `DebtReset` | `WealthChangeStage` | a filer / excluded household carries zero assets |
| `Receipt` | `WealthChangeStage` | `x = (1+r)·a + y(age)·ε` (good); garnished `(1−λ)·y(age)·ε` (excluded) |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | pick next assets `a' ∈ grid`, `c = x − a'` |
| `Readmission` | `MarkovStage` | excluded → good w.p. `ψ` (mean spell `1/ψ`) |

The within-period chain is the `examples/default` chain verbatim — the
credit-market mechanics (gated file/repay choice, debt discharge as a *following*
`WealthChangeStage`, the garnishment haircut, the persistent exclusion spell as
an ordinary `MarkovStage`) are identical. The only life-cycle additions are the
deterministic age-earnings hump `y(age)` and the finite horizon, both supplied
by the driver. The `:status` and `:income` axes are **per-age state**:
`replicate_age` stacks the full six-stage chain per age.

## Why a finite-horizon driver

A life-cycle bankruptcy model is **not** a stationary steady state. A
`ProductStage`'s own `backward!` runs each age-slice independently; it does not
thread age-`(a+1)`'s continuation into age-`a`. So `steady_state.jl` rolls plain
example outer-loop logic:

1. **Backward induction** with a **no-negative-estate terminal condition**:
   with no bequest the period-`(N+1)` value is `0` for `a ≥ 0` and `−∞` for
   `a < 0`, so the last-period household either repays to a non-negative position
   or files. This is a terminal *condition* (an array the driver supplies), not
   a stage. Sweep `a = N…1`, threading the continuation across ages.
2. **Forward cohort simulation.** A unit cohort of newborns (zero wealth,
   ergodic income, good standing) is pushed `a = 1…N`; each age-slice carries
   unit mass.

No new stage, kernel, or per-cell logic — only existing stages' `backward!` /
`forward!` verbs, age-by-age, plus the no-estate terminal array.

## Equilibrium

Prices are **exogenous** (a risk-free unit bond, partial equilibrium): no market
to clear. The solve is the single backward + forward pass.

## Run

```julia
julia --project=. examples/livshits_macgee_tertilt/steady_state.jl
```

With the default calibration the cross-sectional mean assets are positive
(life-cycle saving) with an excluded share ≈ 8%, a filing hazard that peaks
early in life (young, low-income borrowers filing on bad income draws) and
declines with age, and an excluded *stock* share that peaks in early-mid life
(lagged behind the filing flow by the exclusion spell) and decays as agents age
out of debt. Per-age cohort mass is conserved at 1.0 and the value function is
finite on the feasible (non-negative-estate) region.
