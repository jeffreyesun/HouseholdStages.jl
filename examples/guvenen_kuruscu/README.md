# Guvenen–Kuruşçu (2010) — human capital with permanent ability types

A Ben-Porath human-capital life cycle in which households differ in a **permanent
ability type**. Higher-ability types face a lower effort cost of producing human
capital, invest more, and reach higher human capital — widening the lifetime-earnings
distribution. That cross-type widening is the paper's headline.

## Household block — existing stages only

```
Household block (per type) = CapitalInvestmentStage(:h)
Population                  = ⊕_type [ CapitalInvestmentStage(:h) ]
```

The household block **per ability type** is exactly `examples/human_capital`'s
single `CapitalInvestmentStage` on the `:h` axis: from human capital `h` the agent chooses
`h'`, paying a convex effort cost `effort_cost(i; env) = R·(i/a)^{1/γ}` on gross
investment `i = h' − (1−δ)h` and earning `R·h`. The skill price `R` and the
type-specific ability profile `a` arrive through `env`.

## Where the `⊕`-over-types lives: the driver

`CapitalInvestmentStage`'s `production`/`effort_cost` are `(value; env)` closures that read
**only** the operative axis value plus `env` — they cannot read a second `:type`
axis. A literal `⊕` of stages on a `:type` group axis would need a different `env.a`
per group slab within one `backward!`/`forward!` call, which the single shared `env`
cannot provide. So the `⊕`-over-types is realized at the **driver level**: run the
finite-horizon Ben-Porath life cycle once per ability type (each with its own
`env.a` profile, the IDENTICAL backward+forward pattern as `examples/human_capital`),
then stack the per-type cross-sections weighted by population shares. The per-type
`env`-threading is the point — each type genuinely needs its own ability path.

## Driver (example-side, allowed)

`steady_state.jl`:
- `gk_type_life_cycle(k)` — solves one ability type's life cycle (the
  `human_capital` solver, indexed by type `k`).
- `gk_economy()` — loops over types, aggregates the per-type cross-sections by
  population share, and reports the cross-type comparison.

Three ability types (`low`/`mid`/`high`, shares `0.30/0.40/0.30`) differ in their
`(a_young, a_old)` ability profiles.

## Run

```
julia --startup-file=no --project=. examples/guvenen_kuruscu/steady_state.jl
```

Confirmed output: `V` finite for all types, cohort mass conserved (`1.0`) for every
type at every age. Per-type results:

| type | share | a_y→a_o   | peak h | lifetime h̄ | lifetime earn |
|------|-------|-----------|--------|-------------|---------------|
| low  | 0.30  | 0.22→0.03 |  6.53  |   4.23      |   169.3       |
| mid  | 0.40  | 0.32→0.04 | 13.38  |   8.50      |   340.0       |
| high | 0.30  | 0.46→0.06 | 40.36  |  25.97      |  1038.7       |

Higher-ability types invest more and reach higher human capital, giving a
**high/low lifetime-earnings ratio of 6.13** — far wider than within-type
dispersion. Population-weighted lifetime earnings `498.4`. This is the
Guvenen–Kuruşçu headline: permanent ability heterogeneity, amplified through
Ben-Porath investment, is a primary driver of lifetime-earnings inequality.
