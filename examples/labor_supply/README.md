# Aiyagari with endogenous labor (GHH)

An Aiyagari (1994) incomplete-markets economy in which the household jointly chooses
**intensive-margin hours** `n` and savings `b'`. Labor of efficiency `ε` earns `w·ε·n`,
hours carry a disutility `v(n)`, and labor income funds consumption and saving.
References: Chang–Kim (2007), Pijoan-Mas (2006), Heathcote–Storesletten–Violante (2014).

## The ✅ build: closed-form FOC substitution

The catalog marks the *joint* hours/savings solve ◐ (it needs a second choice axis), but
✅ once the intratemporal FOC for hours is solved in closed form and substituted **before**
the savings argmax. This example builds the ✅ version with **GHH preferences**, where the
substitution is exact and especially clean.

GHH felicity is `u(c − v(n))` with `v(n) = ψ·n^{1+1/φ}/(1+1/φ)`. The intratemporal FOC is

```
∂/∂n  u((1+r)b + w·ε·n − b' − v(n)) = u'(·)·(w·ε − v'(n)) = 0
  ⟹  v'(n) = ψ·n^{1/φ} = w·ε  ⟹  n*(w,ε) = (w·ε / ψ)^φ.
```

The GHH optimum `n*` depends **only on the effective wage `w·ε`** — not on wealth,
consumption, or `b'`. So hours are pinned in closed form and folded into the budget, and
the joint problem collapses to the plain Aiyagari spine:

```
IncomeShock ∘ IncomeReceipt(cash = (1+r)b + w·ε·n*) ∘ ConsumptionSavings(u(c − v(n*)))
```

Concretely, the household block (existing stages only):

```julia
shock   = MarkovStage(layout; axis = :income, transition_matrix = P_y)
receipt = IncomeStage(layout;                                  # cash = (1+r)b + w·ε·n*(w·ε)
    wealth_post = (; wealth, income, env) ->
        (1 + env.r) * wealth + env.w * income * n_star(env.w * income))
savings = ConsumptionSavingsStage(layout; β,
    utility      = (cell, c; env) ->                           # GHH composite u(c − v(n*))
        u_crra(c - labor_disutility(n_star(env.w * cell.income)), Val(σ)),
    utility_axes = (:income,))

hh = shock ∘ receipt ∘ savings
```

`n*` and `v(n*)` are constants given the `(w, ε)` a cell faces, so the savings stage is a
standard consumption-savings argmax over a shifted consumption — no extra axis, no coupling.

### Why GHH and not separable `u(c) − v(n)`

With separable CRRA the FOC is `ψ·n^{1/φ} = w·ε·c^{−σ}`, i.e. `n*(c) = (w·ε/ψ)^φ·c^{−σφ}`:
hours depend on consumption, so the labor income that funds `c` itself depends on the
savings choice. That circularity is the ◐ case (the cash-on-hand stops being independent of
the savings argmax). GHH severs the wealth/consumption dependence and gives the exact ✅.

### The ◐ Route-A alternative (not built here)

Keeping hours as a genuine second choice — for non-separable preferences where no closed
`n*` exists — is the `two_asset_hank` auxiliary-choice-axis pattern: an `ArgmaxStage` picking
`n` onto an auxiliary `:hours` axis, a `WealthChangeStage` reading `:hours` to set cash, a
`ForgetfulSumStage` collapsing `:hours`, then `ConsumptionSavingsStage`. It composes from
existing stages too but discretizes hours; the GHH closed form here is exact.

## Equilibrium

General equilibrium with a Cobb-Douglas firm. Both factor markets clear, but the firm's
prices `(r, w)` are pinned by the single capital-labor ratio `κ = K/L`, so the steady state
is a **1-D fixed point in κ**: price κ, solve the household block, read capital supply
`A = ∫ wealth` and effective-labor supply `L_s = ∫ ε·n*` (endogenous, since `n*` rises with
the wage), update `κ → A/L_s`, damp. The outer loop is rolled in `steady_state.jl`; the
per-κ V/Λ solve is delegated to `solve_steady_state_given_env!`.

## Run

```
julia --startup-file=no --project=. examples/labor_supply/steady_state.jl
```

Converges in ~13 outer iterations (~15 s, CPU, `N_w = 300`). Representative output:

```
Converged in 13 outer iterations.
  κ = K/L        = 5.9116
  K (capital)    = 2.2372
  L (eff. labor) = 0.3820
  r              = 0.0355
  w              = 1.2134
  mean wealth    = 2.2372
  mean hours     = 0.3620
  V finite       = true
  hours by productivity state ε (at cleared w = 1.2134):
    ε = 0.60  →  n* = 0.2844
    ε = 1.00  →  n* = 0.3672
    ε = 1.40  →  n* = 0.4344
```

V is finite everywhere, Λ is a conserved stationary distribution (mass 1), and **hours
respond to the productivity state**: `n*` rises monotonically with `ε` at the cleared wage
(0.284 → 0.367 → 0.434), the GHH labor-supply schedule `n* = (w·ε/ψ)^φ`.
