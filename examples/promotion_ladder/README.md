# Promotion / career-effort ladder (convex-cost promotion hazard)

## Mechanism

A savings household on a job ladder pays a **convex effort cost** `c(θ) = θ²/(2κ)` to
raise the hazard `θ` of **climbing one rung up**. Higher rungs pay more, so workers
invest in promotion; the convex cost keeps the promotion intensity interior. An exogenous
job-loss shock occasionally knocks workers down a rung, closing the ladder (income risk +
a non-degenerate cross-section).

State `= (:wealth [continuous log grid], :rung [4-rung ladder, wage values])`.

## The exact `∘` chain

The household block is a pure composition of **existing** library stages (`∘` runs the
LEFT stage first):

```
Promotion ∘ Demotion ∘ Receipt ∘ ConsumptionSavings
```

| Stage | Library stage | Role |
|---|---|---|
| `Promotion`           | `MixingStage` (axis `:rung`)          | Blend `θ ∈ [0,1]` of CLIMB `K_A` (rung i → i+1) and STAY `K_B = I` at cost `θ²/(2κ)`. `θ*(rung) = clamp(κ·(climb − stay), 0, 1)`. |
| `Demotion`            | `MarkovStage` (axis `:rung`)          | Exogenous separation shock: drop one rung w.p. `p_demote`. Closes the ladder. |
| `Receipt`             | `WealthChangeStage` (axis `:wealth`)  | `b ↦ (1+r)b + rung` (the rung VALUE is the wage). |
| `ConsumptionSavings`  | `ConsumptionSavingsStage` (axis `:wealth`) | Pick `b'`; `c = b_in − b'`; CRRA. |

The climb / stay / demotion kernels are row-stochastic matrices built by plain helpers
(`climb_kernel`, `identity_kernel`, `demotion_kernel`) — economic **data** fed to existing
stages, not new stages.

## The §3 contrast: two directions of one hazard control

`Promotion` is `MixingStage` in the **SEARCH direction** — pay a convex cost to
transition UP (`K_A = climb`, `K_B = I`). This is the exact **mirror** of
`examples/insurance`'s `RetentionStage`, which pays a convex cost NOT to transition
(`K_A = I`, `K_B = exit_kernel`). The two examples solve the same closed-form mixing
problem `V = K_B·V + c*(K_A·V − K_B·V)` with the corners swapped:

| | `K_A` (θ=1 corner) | `K_B` (θ=0 corner) | `θ*` interpretation |
|---|---|---|---|
| **Retention** (insurance) | `I` (stay) | exit/loss kernel | survival / coverage probability |
| **Promotion** (this example) | climb kernel | `I` (stay) | promotion intensity |

`θ*` rises with the value gap `climb − stay`: workers on **low rungs**, who have the most
to gain, invest most in promotion; the gap shrinks up the ladder, so `θ*` is decreasing
and pins at 0 at the absorbing top.

## Fidelity note

The convex-cost promotion hazard is a stylized career-ladder/on-the-job-search block (cf.
the job-ladder literature, e.g. Burdett–Mortensen, and human-capital career models). The
brief's minimal **3-stage** chain `Promotion ∘ Receipt ∘ Savings` also solves cleanly, but
with the top rung absorbing and no income risk it absorbs **all** mass at the top rung and
holds ≈ 0 wealth (impatient agents with a deterministic wage hold no precautionary
savings). The 4th stage `Demotion` — an exogenous `MarkovStage` separation shock — closes
the ladder: it restores a non-degenerate rung cross-section and the income risk that makes
the `ConsumptionSavingsStage` spine bite. `Promotion` remains the centerpiece; `Demotion`
is the standard exogenous-shock turnover, fully within the pure-composition rules.

## Status

**Solves cleanly** (`julia --project=. examples/promotion_ladder/steady_state.jl`).
Representative stationary steady state (default params):

```
mass(Λ)        = 1.000000
mean wealth    = 1.1636
mean rung wage = 2.3276
rung distribution (wage ⇒ mass):
  rung 1 (w = 1.00): mass = 0.0031,  θ*(promotion) = 0.4082
  rung 2 (w = 1.50): mass = 0.0382,  θ*(promotion) = 0.4082
  rung 3 (w = 2.00): mass = 0.2591,  θ*(promotion) = 0.2926
  rung 4 (w = 2.50): mass = 0.6996,  θ*(promotion) = 0.0000
θ* range = [0.0000, 0.7222]  (interior ⇒ in (0,1))
```

Mass concentrates up the ladder; the promotion policy `θ*(rung)` is interior in (0,1) and
decreasing in rung — exactly the convex-cost search-direction hazard the example targets.
