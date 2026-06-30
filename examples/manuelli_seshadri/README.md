# Manuelli–Seshadri (2014) — multi-stage human-capital production

Human capital is produced over a **schooling phase** and then an **on-the-job
phase**, using the *same* Ben-Porath technology in both — only the ability and
earnings availability differ by phase.

## Household block (existing stages only)

    Household block = CapitalInvestmentStage(:h)

A single library stage — identical to `examples/human_capital`. From human capital
`h` the agent picks next stock `h'`, paying a convex effort cost on gross
investment `i = h' − (1−δ)h` and earning a flow `earn·h`. There is **no bespoke
household stage** and no new primitive.

## What is driver logic (allowed, expected)

The Manuelli–Seshadri content lives entirely in the **finite-horizon driver** that
threads a phase-specific `env` through the backward/forward sweep
(`env_at_age(t)` in `model.jl`):

- **Schooling phase** (ages `1 … T_school`): no labour earnings (`earn = 0`), high
  learning ability `a_school`, cheap production price. With no opportunity cost of
  working and a long horizon, the agent invests heavily — the **full-time-schooling
  corner**. The corner is produced by the phase `env`, *not* by a new stage.
- **Work phase** (ages `T_school+1 … N_age`): labour earnings `earn·h`, ability
  tapering with age. Investment is now interior and declining.

## Run

    julia --startup-file=no --project=. examples/manuelli_seshadri/steady_state.jl

### Output (confirms it solves)

```
Manuelli–Seshadri multi-stage HC (N_age = 40, T_school = 8, γ = 0.55, δ = 0.04)
  cohort mass (every age)      = 1.000000 … 1.000000   (conserved)
  V finite everywhere          = true
  h at birth   (age  1)        = 1.921
  h at end of school (age  8)  = 14.439
  h at peak    (age  9)        = 16.833
  h at last    (age 40)        = 6.339
  schooling-phase HC growth    = 33.4% per age
  mean investment: school/work = 2.174 / 0.156
```

The headline Manuelli–Seshadri result appears: the overwhelming majority of human
capital is produced during the schooling phase (33%/age growth, ~14× investment
intensity vs the work phase), with the on-the-job phase adding little and
depreciation eventually dominating — a hump peaking just past the end of school.

## Reuse for the rest of the child-HC / multi-phase family

This block — `CapitalInvestmentStage(:h)` driven by a phase schedule — is the **same**
household block needed for the neighbouring literature, differing only in the
length/shape of the `env_at(t)` schedule and (for Daruich) an extra budget term:

- **Caucutt–Lochner** (two-phase early/late childhood HC): same block, a 2-phase
  child schedule.
- **Lee–Seshadri** (many-phase): same block, a longer multi-phase schedule.
- **Daruich** (public transfers): same block plus a transfer term in the budget
  (a `WealthChangeStage` or an `env` transfer), still no new stage.

So this example demonstrates the pattern is reusable for that whole family.
