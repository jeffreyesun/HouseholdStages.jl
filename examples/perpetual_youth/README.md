# Perpetual Youth (Blanchard 1985 / Yaari 1965) — steady state

An incomplete-markets consumption–savings core wrapped in the
demographics composite: a constant death hazard `δ` with mass-preserving
rebirth and actuarially-fair annuities. Survivors earn a mortality
premium on their assets; the dead forfeit their balances to the annuity
pool. The population is stationary at mass ≈ 1.

## The chain

```
IncomeShock ∘ AnnuityReceipt ∘ ConsumptionSavings ∘ ExogenousExit ∘ Entry
```

`∘` is time-ordered: the forward (Λ) sweep runs the left stage first, the
backward (V) sweep runs the right stage first.

- **`IncomeShock`** — `MarkovStage` on the income axis (`P_y`).
- **`AnnuityReceipt`** — `WealthChangeStage`,
  `b ↦ (1+r)/(1−δ)·b + w·y`. The factor `1/(1−δ)` is the actuarially-fair
  annuity return: the dead forfeit their assets, which fund a premium for
  survivors.
- **`ConsumptionSavings`** — `ConsumptionSavingsStage`, pick `b'` on the
  wealth grid; implicit budget `c = b_in − b'`; CRRA (here log) utility.
  The stage's own `β` composes with the exit's survival weighting to give
  an effective continuation discount `β·(1−δ)`.
- **`ExogenousExit`** — survival `s = 1−δ`. Backward
  `V ↦ s·V + (1−s)·bequest`; forward `Λ ↦ s·Λ` (deaths leak mass out).
  Requires the layout to declare `:exiting => Discrete([0])` (size 1); the
  composite grows it `1 → 2` internally.
- **`Entry`** — `EntryStage`, additive newborn source `Λ ↦ Λ + g`.
  Newborns enter at zero wealth with the ergodic income draw; `Σg = δ`.

```julia
hh = shock ∘ receipt ∘ savings ∘ exit ∘ entry
```

## The Blanchard–Yaari cancellation

The savings Euler equation discounts the continuation by `β·s` (the
ExogenousExit weighting), while the annuity grosses assets up by `1/s`.
The effective return–discount product is therefore
`β·s·(1+r)/s = β(1+r)` — exactly the standard Aiyagari condition
`β(1+r) < 1` for bounded wealth accumulation. The mortality premium and
the survival discount cancel cleanly. (At the calibration below,
`β(1+r) = 0.96 × 1.03 = 0.9888 < 1`.)

## Stationary population

The forward mass map collapses to `M_{t+1} = (1−δ)·M_t + Σg`. With
`Σg = δ` (a newborn inflow exactly replacing deaths) the fixed point is
`M* = δ/(1−(1−δ)) = 1`. The newborn distribution `g` puts mass `δ` at
zero wealth, split across the income states by the ergodic distribution
`π` of `P_y`:

```julia
π = ergodic_distribution(p.P_y)
g = zeros(p.N_w, n_y, 1)
g[1, :, 1] .= p.δ .* π
entry = EntryStage(layout; entry = g)
```

## The `bequest` choice

`ExogenousExit` requires a `bequest` (the value of death). Here death is
*exogenous* — households do not choose to die — so the bequest level
enters `V` only as the uniform additive term `(1−δ)·bequest` in
`V_start = s·V_end + (1−s)·bequest`. A uniform additive shift in the
continuation does not change the argmax in the savings step, so **the
bequest level affects the level of `V` but not the policy or the
stationary distribution.** With log utility (`σ = 1`) the natural
normalization is `bequest = 0.0` (no warm-glow); `V` stays finite
everywhere. A warm-glow `θ·log(1+wealth)` would be the place to add a
genuine bequest motive if one wanted it to bite.

## Calibration and result

Default `PerpetualYouthParams`: β = 0.96, σ = 1.0 (log), δ = 0.05;
3-state income on `[0.5, 1.0, 1.5]` with a sticky transition; `N_w = 120`
log-spaced wealth points on `[0, 80]`; `bequest = 0.0`. Partial
equilibrium at `r = 0.03`, `w = 1.0`.

```
annuity gross return  = 1.0842   (= (1+r)/(1−δ))
population mass  ΣΛ   = 1.000000
mean wealth E[w]      = 1.2177
V finite everywhere   = true
```

A single `solve_steady_state_given_env!` call (no market to clear): the
library runs V backward to a fixed point, then Λ forward to its
stationary distribution, then evaluates the attached moments
(`A_total = ∫ wealth dΛ`, `pop = ∫ dΛ`).

## Run

```bash
julia --project=. examples/perpetual_youth/steady_state.jl
```

About 11 seconds including first-call compilation; the solve itself is
sub-second at `N_w = 120`.

## Files

- `model.jl` — parameters, ergodic-income helper, layout, the five-stage
  chain, newborn source `g`.
- `steady_state.jl` — the single partial-equilibrium solve and the
  reported moments.
