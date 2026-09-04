# CARA–normal portfolio benchmark — the textbook check on `GaussianLoadingStage`

The textbook **check** that `GaussianLoadingStage` — in its canonical portfolio reading (anchor =
`R_f`, increment = the Gaussian excess return) — recovers the classic
mean-variance share

```
θ* ≈ (μ − R_f) / (γ · σ²)
```

with `μ = E[R̃]` the mean gross risky return, `σ² = Var(R̃)`, `γ` the risk-aversion coefficient.
"CARA–normal" is the textbook packaging (CARA utility + normal returns gives this rule exactly) — and
the stage's return contract **is** that setup natively: the excess return is a (±8σ-truncated)
Gaussian and θ is continuous, so the benchmark's distributional assumption holds by construction
rather than by approximation. The package felicity is CRRA `u_crra(c, Val(γ))`,
for which the same rule is the **Samuelson–Merton myopic share** — and in the one-period problem that
share is **exactly wealth-independent** for CRRA (factor `W^{1-γ}` out of
`E[(W·(R_f + θ·excess))^{1-γ}]` and no `W` remains in the θ-objective).

## Household block

A **single existing library stage** — no bespoke household stage:

```
Portfolio = GaussianLoadingStage(:wealth; loading_bounds = (0.0, 1.5))
```

The benchmark is the **one-period** mean-variance problem. With terminal value
`V_end(w) = u_crra(w, Val(γ))`, one `backward!` of `GaussianLoadingStage` solves
`max_θ E[u(b'·(R_f + θ·(μ_x + σ_x·Z)))]` per wealth cell and seats `θ*(x)`. This isolates the **static**
portfolio rule from the multi-period human-wealth / Merton-hedging effects a full consumption-savings
steady state would layer on (those tilt the young toward equity — the Cocco–Gomes channel — and are a
different object from the textbook static share). The excess moments are matched to a two-point
gamble `premium ± d`, so `μ_x = premium` and `σ_x = d` (`Var = d²` exactly) at `p_up = ½`.

## The check, and the gaps

With the default calibration (`γ = 2`, premium `0.01`, `d = 0.10` ⇒ `σ² = 0.01`, formula `θ* = 0.5`):

- **Interior and wealth-independent.** Over the interior band (cells whose landing band, out to a
  few sd, stays on-grid), seated `θ*(x) ≈ 0.5087`, flat to `sd ≈ 2e-5` — the wealth-independence
  the CRRA one-period rule predicts, held to five decimal places.
- **Matches the formula** to ≈ 1.7% (`0.5087` vs `0.5`).
- **Grid edges corner out** (`θ* = 0` at the top, `θ* = 1.5` at the bottom): a near-edge landing band
  spills outside `[w_min, w_max]` and is clamped, breaking the homogeneity. These cells are excluded
  from the interior check — they are a grid artifact, not a portfolio result.

Two gaps separate the seated share from the closed form. Neither is a discretization of the share
itself: θ is continuous, solved by scan + Newton.

1. **Log-grid interpolation** — the continuation is interpolated at off-node landings on a log-spaced
   wealth grid. Under the smooth Gaussian row this averages out almost completely: it shows up only
   in the interior dispersion (≈ 6e-5 at `N_w = 80`, shrinking with refinement), not in the mean.
2. **Higher-moment / CRRA≈CARA error** — the formula is a 2nd-order mean-variance approximation of
   the CRRA objective; the exact CRRA optimum against the Gaussian (±8σ-truncated) return differs
   by a fixed, `N_w`-independent amount.

**Grid refinement** isolates these: as `N_w` grows `80 → 160 → 320 → 640`, the interior mean sits at
`0.5087` throughout while the dispersion shrinks `6e-5 → 2e-5 → 4e-6 → 1e-6`. That split — a mean
that does not move with the grid, a dispersion that vanishes with it — is the diagnostic: (1) is
purely numerical and refines away, while the residual `+0.0087` (≈ 1.7%) is the higher-moment
correction (2), a genuine model gap.

## Run

```julia
julia --project=. examples/cara_normal/steady_state.jl
```

Reports the seated `θ*(x)` range over the full grid and the interior band, the wealth-independence
check, the gap to the closed-form `(μ − R_f)/(γσ²)`, and the grid-refinement trace.
