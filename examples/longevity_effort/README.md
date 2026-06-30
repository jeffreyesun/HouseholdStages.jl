# Longevity effort — survival as a retention choice

Grossman (1972) demand-for-health / Pijoan-Mas & Ríos-Rull (2014) survival-effort
model, as a **life-cycle household block built from existing library stages only**.
The household pays a convex **effort cost to RAISE its own per-period survival
probability** — the "value of life" margin. Survival itself is the retention
choice.

## Mechanism

Each age the agent chooses a survival probability `θ ∈ [0,1]` at convex effort
cost `c(θ) = θ²/(2κ)` (in **utils**). With probability `θ` it lives through the
age (and earns the value of being alive plus consumption); with probability
`1−θ` it dies. The closed form

```
V_alive_start = (death value) + c*(alive value − death value),   θ*(w) = clamp(κ·(V_alive − V_dead), 0, 1)
```

makes survival effort rise with the **continuation value of being alive**. Since
the dead earn zero flow (their value is pinned at 0), `θ*` is driven by `V_alive`
itself: richer agents (more consumption value) and younger agents (more years of
life left to protect) buy more survival. That is the value-of-life mechanism.

## The exact chain

```
Survival ∘ Receipt ∘ ConsumptionSavings
```

(`∘` runs the **left** stage first, in time order.) State = `(:wealth, :alive)`
with `:alive ∈ {:alive, :dead}`.

| Stage | Library stage | Role |
|---|---|---|
| `Survival` | `RetentionStage(axis = :alive)` | `K_A = I` (stay alive), `K_B =` certain-death kernel (alive → dead w.p. 1, dead absorbing). The blended alive row is `[θ, 1−θ]`, so the retention weight `θ` **is** the survival probability, bought at convex utils effort cost `c(θ)=θ²/(2κ)`. |
| `Receipt` | `WealthChangeStage` | `w ↦ (1+r)·w + y`: return on assets plus labour income. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | picks next-period wealth; `c = x − w'`, flow `b̄ + u_crra(c)` while alive, **zero flow when dead** (`cell.alive == :dead`), which pins the death-state value at 0. |

`RetentionStage` is exactly the `examples/insurance` `MixingStage` machinery, but
the retention axis is `:alive` (survival) rather than `:wealth` (the asset stock).
It is distinct from `examples/health`, where survival is a `CapitalInvestmentStage`
on a health stock plus a health-dependent `MarkovStage` — there survival is bought
*indirectly* through a stock; here it is the *direct* retention choice.

The two plain helper functions (`death_kernel`, and the dead-zeroing utility
closure) build **data** fed to existing stages — no bespoke household stage, no
new kernel/`backward!`. This obeys the §3 dogfooding rule that the household block
is a pure `∘` chain of exported stages.

## Why finite-horizon (life cycle), not stationary

Mortality leaks mass into the absorbing dead state with no birth source, so there
is no nondegenerate stationary **alive**-mass (the `Λ' = K·Λ + M·g` mass-injection
gap). The finite-horizon cohort sidesteps this honestly: a cohort born alive at
`w0` decays along its chosen survival curve over the life cycle. Total grid mass is
conserved (it accumulates in `dead`). The driver in `steady_state.jl` mirrors
`examples/health/steady_state.jl`: backward-induct `V` over ages (terminal `V = 0`),
then forward-push a point-mass cohort, capturing the seated survival policy at each
age.

## Fidelity caveat — utils cost, NOT a budget drain

`RetentionStage`'s effort cost `c(θ)` is a **utils** (effort-disutility) cost,
entering `V` additively through the Fenchel conjugate. It is **not** a resource
cost competing with consumption out of the budget. So in this model wealth affects
survival **only through the continuation value of being alive** (richer ⇒ higher
`V_alive` ⇒ more survival effort), never through a medical-spending budget drain.

The resource-cost version — medical expenditure paid out of cash-on-hand, which
*does* trade off against consumption — is **not expressible by `RetentionStage`**.
That requires a `CapitalInvestmentStage` on a health/survival stock with a
convex resource cost on gross investment (see `examples/health`). The two readings
are economically different (utils effort vs. resource spending); this example is
the utils-effort reading, stated honestly.

A second normalization point: a per-period **flow value of being alive** `b̄` is
added to the CRRA flow (`b̄ + u_crra(c)`). This is the standard value-of-life
normalization (Hall–Jones, Rosen): it keeps `V_alive > V_dead = 0` so the survival
margin is well-posed under CRRA with `σ > 1` (where `u_crra < 0`). Without it the
alive value can be negative and survival effort collapses to zero.

## Status

Solves cleanly (`julia --project=. examples/longevity_effort/steady_state.jl`).
At the default calibration (`κ = 0.05`, `b̄ = 11`, `σ = 2`, `N_age = 40`,
`N_w = 80`):

- Total grid mass conserved at `1.000000` across all ages (alive + dead); `V`
  finite everywhere.
- Survival curve declines over the life cycle (`1.00` → `0.05` → `0.0005`); life
  expectancy `Σ_t survival ≈ 8.3` ages.
- **Value-of-life mechanism confirmed:** age-1 survival effort `θ*` rises with
  wealth (`0.83` for the poorest → `1.0` for the richest, who fully insure their
  survival), and at fixed wealth falls with age (`0.93` young → `0.54` at the end
  of life — fewer years left to protect).
