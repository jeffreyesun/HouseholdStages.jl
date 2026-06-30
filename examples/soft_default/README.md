# Soft default — a convex-cost probability-of-default hazard

A consumer-default / debt model where the household **smoothly scales its
probability of default** at a convex cost, built from existing library stages
only. This is the **soft** reading of the default margin; the hard, discrete
repay-vs-default stopping problem (Eaton–Gersovitz / Arellano) is §5's
`DefaultStage` (see `examples/default`).

## Mechanism

Each period the household chooses a default probability `θ ∈ [0,1]` at convex cost
`c(θ) = θ²/(2κ)`. With probability `θ` it **defaults**: its balance sheet resets to
a clean slate (zero wealth, debt discharged). With probability `1−θ` it **keeps**
the balance sheet (repays). The closed form

```
V = (keep value) + c*(reset value − keep value),   θ*(w) = clamp(κ·(V(0) − V(w)), 0, 1)
```

makes default rise exactly where the clean slate beats keeping the balance sheet —
i.e. for **deeply indebted** agents (`w < 0`), so default concentrates in the debt
region. Agents with positive assets never default (`V(0) < V(w)` ⇒ `θ* = 0`).

## The exact chain

```
IncomeShock ∘ DefaultChoice ∘ Receipt ∘ ConsumptionSavings
```

(`∘` runs the **left** stage first, in time order.) State = `(:wealth, :income)`,
with the wealth grid spanning a debt region (`w_min < 0 < w_max`).

| Stage | Library stage | Role |
|---|---|---|
| `IncomeShock` | `MarkovStage(axis = :income)` | the endowment process. |
| `DefaultChoice` | `MixingStage(axis = :wealth)` | blends `K_A =` reset kernel (every wealth cell → the zero-wealth grid point: clean slate) and `K_B = I` (keep / repay) at convex cost. The weight on the reset corner `θ` **is** the default probability. |
| `Receipt` | `WealthChangeStage` | `w ↦ (1+r)·w + income`: interest on the (post-default) balance sheet plus income; can push agents into debt. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | picks next-period wealth (the floor `w_min < 0` is the borrowing limit); `c = x − w'`, CRRA utility. Borrowing is allowed; the calibration keeps `c > 0` on the active region. |

The two plain helpers (`reset_kernel`, `identity_kernel`) build **row-stochastic
data** fed to `MixingStage` — same construction as `examples/insurance`'s
`loss_kernel`, but the reset destination is a fixed value (0) rather than
`loss_factor·w`. No bespoke household stage, no new kernel/`backward!`: the block
is a pure `∘` chain of exported stages, per the §3 dogfooding rule.

## Why `MixingStage`, not `LogitEndogenousExit`

There are two library readings of a soft default:

- **`LogitEndogenousExit`** — default as a soft **exit**: the defaulting mass
  *leaves* the population (a bequest is paid). Mass is **not** conserved — a
  default-rate share leaks out each period, which would need a re-entry / birth
  source to stay stationary.
- **`MixingStage` with a reset kernel** (this example) — default as a clean-slate
  **reset**: the defaulting mass *stays* in the population at zero wealth. Because
  the reset kernel is row-stochastic, total wealth-grid mass is **conserved** by
  construction.

For consumer/sovereign default the agent persists after discharging its debt, so
the reset reading is the right one, and mass conservation makes it the cleaner
stationary object. This example takes the reset reading.

## Fidelity note — default crowds out precautionary saving

A within-period reset-to-0 default is a **perfect downside wealth floor**: no agent
needs to self-insure below zero, because default bails it out at low cost. This
crowds out the standard precautionary saving motive, so the stationary distribution
is debt-heavy unless agents have a standalone saving motive. The default
calibration uses `β·(1+r) ≈ 0.9996` (patient, but still `< 1` for stationarity) so
that a consumption-smoothing saving motive coexists with the default option,
yielding a distribution that genuinely **spans debt and assets** — which is what
makes "default concentrates in the debt region" a meaningful statement rather than
a tautology. This crowding-out is an economically correct feature of cheap reset
default, not a numerical artifact.

Stabilization knobs (per the §3 brief): the default rate, the debt/asset spread,
and VFI convergence are tuned through `κ` (default cost), `w_min`/`w_max` (the
borrowing limit and grid top), `β`, and `r`. Too-cheap default (`κ` large) collapses
to a degenerate borrow-to-the-limit-then-wipe money pump (all mass at `w_min`,
`θ = 1`); the calibration below avoids it.

## Status

Solves cleanly (`julia --project=. examples/soft_default/steady_state.jl`). At the
default calibration (`β = 0.98`, `κ = 0.2`, `σ = 2`, `r = 0.02`,
`w ∈ [−1, 8]`, `N_w = 120`):

- `mass(Λ) = 1.000000` (mass-conserving); no mass piles at the grid top.
- Distribution spans the grid: ~`69%` in debt (`w < 0`), ~`31%` holding positive
  assets; mean wealth ≈ `−0.06`.
- Average default probability `θ̄ ≈ 0.090`, **interior** (`θ* ∈ [0, 0.23]`, never
  hits the corners 0/1 degenerately).
- **Default concentrates in debt:** mass-weighted `θ̄` is `0.131` on the debt slice
  vs `0.000` on the asset slice.
