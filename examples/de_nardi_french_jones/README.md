# De Nardi–French–Jones (2010) — medical expenses & mortality in old age

Retired elderly save against two intertwined old-age risks: uncertain
out-of-pocket **medical expenses** (high and rising near death) and uncertain
**mortality** (health- and age-dependent). The central finding: the
medical-expense risk that spikes near the end of life rationalises why the
elderly **dissave slowly** — they hold a precautionary buffer against a costly,
uncertain death (reinforced here by a bequest motive).

## The household block (dogfooding)

A single existing library object — `replicate_age` of a `∘`-chain of **exported
stages only**:

```julia
replicate_age( MarkovStage(:health)            # good/bad health dynamics
             ∘ WealthChangeStage(medical)       # x = max((1+r)w + pension − m(health,age), c_floor)
             ∘ ConsumptionSavingsStage(:wealth) # pick next assets a', c = x − a'
             ∘ ExogenousExit(survival, bequest), # health-&-age-dependent mortality
             N; axis = :age )
```

- **Stochastic medical expense.** `WealthChangeStage` maps wealth to cash-on-hand
  net of `m(health, age) = med_base[health]·(1 + med_growth·(age−1))`. It is
  stochastic because `health` follows a Markov chain and the expense reads BOTH
  the (random) health state and the age. The `c_floor` is a means-tested
  consumption floor (Medicaid).
- **Health-and-age-dependent mortality** is exactly `ExogenousExit`'s `survival`
  dep closure `s(health, env.age)`. The dying `(1−s)` fraction LEAVES the cohort
  each age, so the cohort shrinks with age — correct here: a retired cohort dies
  out, and there is no entry.
- **Identical age-slices.** `replicate_age` requires identical slices, so both the
  medical expense and the survival hazard read the age through the per-age `env`.
- **No new stages, no `src/` edits.** The finite-horizon sweep is example-side.

## The driver (example-side)

`steady_state.jl` runs a single **backward sweep** `a = N…1` (threading each
age-(a+1)'s continuation into age-a's `replicate_age` component — a `ProductStage`
does not do this) followed by a **forward cohort simulation** `a = 1…N` in which
the exit composite leaks the dying fraction each age. Newly-retired agents start
at assets `w_init` with the stationary health distribution.

## Run

```
julia --project=. examples/de_nardi_french_jones/steady_state.jl
```

Representative output (default calibration, `N = 20`, `N_w = 100`):

```
V finite                : true
surviving mass age 1→N  : 1.0000 → 0.0159
mean wealth age 1→N     : 7.89 → 0.51
asset half-life reached : age 11
```

The survivor wealth profile declines **slowly** — ~80% of initial assets retained
at age 5, ~51% at age 10 — driven by the precautionary buffer against medical
risk plus the bequest motive, while the cohort thins out via health/age mortality.

## Moments

- **mean wealth by age** among survivors (the slow-dissaving asset profile),
- **surviving mass by age** (the cohort's mortality curve),
- the age-20 medical-expense and survival levels by health, for reference.
