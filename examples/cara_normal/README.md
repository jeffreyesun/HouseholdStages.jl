# CARA–normal portfolio benchmark — the textbook check on `MeanVarianceStage`

The textbook **check** that `MeanVarianceStage`'s streaming `(max, +)` recovers the classic
mean-variance share

```
θ* ≈ (μ − R_f) / (γ · σ²)
```

with `μ = E[R̃]` the mean gross risky return, `σ² = Var(R̃)`, `γ` the risk-aversion coefficient.
"CARA–normal" is the textbook packaging (CARA utility + normal returns gives this rule exactly); the
package felicity is CRRA `u_crra(c, Val(γ))`, for which the same rule is the **Samuelson–Merton myopic
share** — and in the one-period problem that share is **exactly wealth-independent** for CRRA (factor
`W^{1-γ}` out of `E[(W·(R_f + θ·excess))^{1-γ}]` and no `W` remains in the θ-objective).

## Household block

A **single existing library stage** — no bespoke household stage:

```
Portfolio = MeanVarianceStage(:wealth, fine share grid)
```

The benchmark is the **one-period** mean-variance problem. With terminal value
`V_end(w) = u_crra(w, Val(γ))`, one `backward!` of `MeanVarianceStage` solves
`max_θ E[u(b'·(R_f + θ·(R_k − R_f)))]` per wealth cell and seats `θ*(x)`. This isolates the **static**
portfolio rule from the multi-period human-wealth / Merton-hedging effects a full consumption-savings
steady state would layer on (those tilt the young toward equity — the Cocco–Gomes channel — and are a
different object from the textbook static share). The risky leg is a symmetric two-point gamble
`μ ± d`, so `Var = d²` exactly.

## The check, and the gaps

With the default calibration (`γ = 2`, premium `0.01`, `d = 0.10` ⇒ `σ² = 0.01`, formula `θ* = 0.5`):

- **Interior and wealth-independent.** Over the interior band (cells whose worst/best-case landing
  stays on-grid), seated `θ*(x) ≈ 0.53`, flat (sd ≈ 0.03 ≈ a few share-grid steps) — the
  wealth-independence the CRRA one-period rule predicts.
- **Matches the formula** to ≈ 5% at the default grid (`0.526` vs `0.5`).
- **Grid edges corner out** (`θ* = 0` at the top, `θ* = 1.5` at the bottom): a near-edge landing
  `w·(R_f ± θd)` falls outside `[w_min, w_max]` and is clamped, breaking the homogeneity. These cells
  are excluded from the interior check — they are a grid artifact, not a portfolio result.

Three gaps separate the seated share from the closed form:

1. **Share-grid step** `Δθ = 0.005` — the policy is resolved only to the grid.
2. **Log-grid interpolation** — the continuation is interpolated at off-node landings on a log-spaced
   wealth grid; this is the dominant, `N_w`-sensitive error.
3. **Higher-moment / CRRA≈CARA error** — the formula is a 2nd-order mean-variance approximation, but
   the risky leg is a two-point gamble (not normal), so the exact CRRA optimum differs by a fixed,
   `N_w`-independent amount.

**Grid refinement** isolates these: as `N_w` grows `80 → 160 → 320 → 640`, the interior mean settles
`0.359 → 0.526 → 0.514 → 0.513` and the dispersion shrinks `0.090 → 0.034 → 0.024 → 0.013`. The
interpolation error (2) vanishes; the residual ≈ `0.013` (≈ 2.7%) is the higher-moment correction
(3) — the genuine model gap, not a numerical one.

## Run

```julia
julia --project=. examples/cara_normal/steady_state.jl
```

Reports the seated `θ*(x)` range over the full grid and the interior band, the wealth-independence
check, the gap to the closed-form `(μ − R_f)/(γσ²)`, and the grid-refinement trace.
