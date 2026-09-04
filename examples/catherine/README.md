# Catherine (2022) — countercyclical risk + life-cycle portfolio

Countercyclical idiosyncratic income risk (higher variance in recessions)
makes the **human-wealth hedge state-dependent**. The young hold most of their
wealth as human capital, so they bear the brunt of cyclical income risk: their
precautionary saving and their financial portfolio both respond to the
aggregate state. This example overlays the countercyclical income process of
`examples/countercyclical_risk` onto the CGM life-cycle portfolio chain of
`examples/cocco_gomes_maenhout`.

## The household block (existing stages only)

The CGM four-stage chain, with the income shock's transition swapped for the
countercyclical **env-closure**:

```
replicate_age(
    MarkovStage(:income; transition_matrix = (; env) -> catherine_T(env.z))
        ∘ Receipt ∘ ConsumptionSavings ∘ Portfolio,
    N; axis = :age)
```

| stage | library object | role |
|---|---|---|
| `IncomeShock` | `MarkovStage(:income)` | **env-dependent** transition `T(env.z)`: wider innovation std in recessions |
| `Receipt` | `WealthChangeStage` | cash-on-hand `x = b + y(age)·ε` |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | picks invested wealth `b'` |
| `Portfolio` | `GaussianLoadingStage` (portfolio reading: anchor = `R_f`, increment = excess return) | picks the continuous risky share `θ ∈ [0, 1]`; next wealth `b'·(R_f + θ·(μ_x + σ_x·Z))`, the Gaussian excess moment-matched to the equity calibration |

**No bespoke stage.** This is the CGM chain with the `countercyclical_risk`
env-closure (a Tauchen fill whose conditional variance rises in recessions, at
fixed persistence) bolted onto its income shock. The income `MarkovStage`
re-seats its transition whenever `env.z` changes.

## Solve

In a life-cycle solve the aggregate state `z` is fixed over the cohort's life,
so we run the whole backward+forward sweep **twice** on the same household
object — every age facing the boom transition, then every age facing the
recession transition — and compare. (No fixed-point iteration: a life-cycle
solve is one backward sweep plus one forward cohort simulation.)

## Results (`σ = 4`, premium ≈ 5%, σ_rec = 0.24 = 2 × σ_boom)

```
                       boom     recession    Δ
mean wealth (x-sec)   1.645      1.897     +15.3%
θ* (mid-career)       ~0.94      ~0.88
Δθ (max, ages 15-18)             ≈ −0.07
```

- **More precautionary self-insurance in recession.** Mean wealth is 15.3%
  higher in the recession solve, and recession financial wealth exceeds boom
  wealth at every age — the wider income innovations strengthen the
  precautionary motive.
- **Lower risky share in recession** (Δθ < 0 at every interior age; the
  cash-constrained young are pinned at the equity cap in both states).
  Countercyclical income risk makes human wealth riskier and lifts financial
  wealth, both pushing the financial equity share down. The gap is largest in
  mid-career (ages 14-18, Δθ ≈ −0.05 to −0.07), exactly where workers carry
  the most cyclical human-capital risk.
- Both profiles inherit the CGM age-decline of the share toward the Merton
  interior (≈ 0.28).

That the same household object delivers two materially different θ*(age)
profiles and wealth levels demonstrates the env-closure genuinely re-seats the
income transition when `env.z` changes.

## Cost

A life-cycle solve is a single backward+forward sweep with no fixed point, so
the two aggregate states together run in well under a minute — cheap enough
that the full life-cycle version needs no stationary approximation.

The newborn financial endowment (`w0_init`, see the CGM README) is carried over
for the same reason — the package's `GaussianLoadingStage` allocates over financial
wealth, so a zero-wealth newborn de-risks rather than holding the human-wealth-
tilted full-equity share.

## Run

```
julia --startup-file=no --project=. examples/catherine/steady_state.jl
```
