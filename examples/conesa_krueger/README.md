# Conesa–Krueger (1999) — OLG with PAYG social security

A finite-horizon OLG household à la **Conesa & Krueger (1999)**, "Social
Security Reform with Heterogeneous Agents": an `N`-period household that saves
against a hump-shaped earnings profile and persistent income risk, embedded in
an unfunded pay-as-you-go (PAYG) social security system. Working-age agents pay
a flat payroll tax `τ` on labor earnings; retirees draw a flat benefit `b`. The
household block is the **same `replicate_age(…)` of existing stages** as
`examples/life_cycle` — **no bespoke household stage** — with the social-security
wedge riding the receipt closure.

## Household block

```
replicate_age( IncomeShock ∘ Receipt+SS ∘ ConsumptionSavings , N; axis = :age )
```

| stage | library stage | does |
|---|---|---|
| `replicate_age(…, N; axis=:age)` | `ProductStage` | stack `N` age-slices on the `:age` axis |
| `IncomeShock` | `MarkovStage` | persistent earnings risk on the `:income` axis |
| `Receipt+SS` | `WealthChangeStage` | `b ↦ (1+r)·b + (1−τ)·y(age)·ε + benefit` |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | pick next wealth `b'`, `c = x − b'`, CRRA |

The social-security tax/benefit is **not a new stage** — it is the receipt
`WealthChangeStage` closure reading the net wage `(1−τ)·y(age)` and the benefit
`b` out of the per-age `env`. A worker (`age ≤ retire_age`) pays the payroll tax
and earns the hump wage; a retiree has zero labor earnings and draws the flat
benefit. Age-dependence rides `env` because the `:age` axis is a size-1
singleton inside each `replicate_age` component (exactly how `life_cycle`
threads its hump `y(age)`).

## Why a finite-horizon driver, plus a PAYG balance loop

An OLG model is **not** a stationary steady state. A `ProductStage`'s own
`backward!` runs each age-slice independently; it does not thread age-`(a+1)`'s
continuation into age-`a`. So `steady_state.jl` rolls (plain example outer-loop
logic, the same status as a tatonnement):

1. **Backward induction.** With no bequest (`V_{N+1} = 0`), sweep `a = N…1`,
   feeding age-`(a+1)`'s continuation into age-`a`'s component.
2. **Forward cohort simulation.** A unit cohort of newborns (zero wealth,
   ergodic income) is pushed `a = 1…N`; each age-slice carries unit mass.
3. **PAYG budget balance.** The unfunded benefit `b` is pinned by
   `b · (#retired cohorts) = τ · (aggregate labor earnings)`. Because `b` enters
   retirees' receipt, this is a fixed point: re-run the backward+forward pass,
   update `b = τ·E / R`, iterate to convergence. (Labor supply is inelastic and
   earnings are exogenous, so `E` is independent of `b` and the loop converges
   essentially one-shot — but it is a genuine balance condition, run as written.)
   Pass `balance = false` to fix `b` at `b0` for pure partial equilibrium.

No new stage, kernel, or per-cell logic is introduced — only existing stages'
`backward!` / `forward!` verbs, age-by-age, plus the SS clearing loop.

## Equilibrium

Prices `r` and the earnings profile `y(age)` are **exogenous** (partial
equilibrium); the **PAYG social-security system clears** (benefits = taxes).

## Run

```julia
julia --project=. examples/conesa_krueger/steady_state.jl
```

With the default calibration (`N = 24`, retire at 16, `τ = 0.10`) the benefit
balances at a replacement rate `b/peak-wage ≈ 0.18`, total benefits paid equal
total payroll taxes, per-age cohort mass is conserved at 1.0, and the wealth
profile is the textbook OLG hump: zero at birth, accumulation through working
life, a peak just after retirement, and decumulation thereafter.
