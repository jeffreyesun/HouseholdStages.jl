# De Nardi (2004) — bequests, inheritance & wealth concentration

Finite-lived agents save over the life cycle and leave **bequests** (voluntary
warm-glow + accidental) when they die. Inherited wealth links one generation to
the next into a **dynasty**, fattening the top tail of the stationary wealth
distribution relative to a model without bequests.

## The household block (dogfooding)

A single existing library object — `replicate_age` of a `∘`-chain of **exported
stages only**:

```julia
replicate_age( MarkovStage(:income)            # persistent earnings risk
             ∘ WealthChangeStage(receipt)       # cash-on-hand (1+r)b + y(age)·ε
             ∘ ConsumptionSavingsStage(:wealth) # pick next assets a', c = x − a'
             ∘ ExogenousExit(survival, bequest), # mortality + value of dying
             N; axis = :age )
```

- **Exit at every age.** `replicate_age` requires identical age-slices, so
  `ExogenousExit` applies at *every* age with an env-dependent survival closure
  `s(env.age)` (accidental bequests arise at every age, not only the terminal
  one). All slices are the same chain; the age enters only through the per-age
  `env`.
- **Warm-glow bequest.** The exit composite's required `bequest` field is
  `b(a') = φ·u_crra(a' + κ)` — the joy-of-giving value of the assets `a'` carried
  into death. `κ` makes bequests a luxury good (only the rich leave them),
  De Nardi's mechanism for the fat tail. This raises the continuation value
  with `a'`, so the bequest motive shows up as extra saving.
- **No new stages.** No `@definestage`, no custom kernel/forward!/backward!, no
  `src/` edits. Everything below is example-side driver code.

## The dynastic closure (example-side)

`steady_state.jl` runs two outer loops over the block:

1. **Finite-horizon backward sweep** `a = N…1`, threading each age-(a+1)'s
   continuation into age-a's `replicate_age` component (a `ProductStage` does not
   thread continuations across ages — that wiring is the example's job). Done
   once; policies do not depend on the dynastic distribution.
2. **Dynastic fixed point.** The dynasty links the assets the dying bequeath to
   the assets newborns inherit. We iterate the newborn inherited-wealth
   distribution `g`: seed a unit cohort with `g` through an **`EntryStage(entry =
   g)`** (the dynastic link is exactly `EntryStage`'s `g`), push it forward
   `a = 1…N` (the exit composite leaks the dying fraction each age), tally the
   cross-age distribution of bequeathed assets, and set the next `g` to it.

Since survival `s(a)` is scalar per age, exit scales the cohort uniformly: the
post-savings mass at age `a` is `Λ_next / s(a)`, of which `(1−s(a))` dies and
bequeaths its assets `a'`. The final age-N cohort all die (no age N+1), so its
whole post-savings mass is bequeathed; the bequest masses then telescope to 1, so
**one death funds one newborn** — a genuine cross-generation stationary
distribution. This is the **full** dynastic closure (not a single-generation
simplification).

## Run

```
julia --project=. examples/de_nardi_bequests/steady_state.jl
```

Representative output (default calibration, `N = 20`, `N_w = 100`):

```
dynastic closure        : 8 iters, final |Δg| = 7.6e-08
V finite                : true
newborn mass / total pop: 1.0000 / 14.86
surviving mass age 1→N  : 1.0000 → 0.3744
mean wealth (pooled)    : 0.765
asset profile 1/peak/N  : 0.605 / 1.873 (age 15) / 0.304
top 10% wealth share    : 0.344
top 1%  wealth share    : 0.048
```

## Moments

- **mean wealth** (pooled cross-section over living ages),
- **top-decile / top-percentile wealth share** (the concentration the bequest
  channel fattens),
- **total mass** (pooled over the shrinking cohort) and the surviving-mass
  profile.
