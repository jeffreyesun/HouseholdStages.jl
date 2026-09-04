# Portfolio choice — incomplete-markets steady state

An Aiyagari savings problem with a Merton / Cocco–Gomes–Maenhout **portfolio decision**: each period
the household chooses how much to save *and* what risky share to hold. The within-period problem is
**four existing `HouseholdStages` stages, no bespoke household stage** — the demonstration this
example exists for (Part 3: literature household blocks expressible from the library alone).

## Household block

In time order:

```
IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Portfolio
```

| stage | library stage | does |
|---|---|---|
| `IncomeShock` | `MarkovStage` | draw next income on the `:income` axis |
| `Receipt` | `WealthChangeStage` | `a ↦ a + w·y` (cash-on-hand) |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | pick next financial wealth `b'`, `c = x − b'`, CRRA utility |
| `Portfolio` | `GaussianLoadingStage` | pick the continuous risky share `θ ∈ [0, 1]`; next wealth `b'·(R_f + θ·(μ_x + σ_x·Z))` |

`GaussianLoadingStage` — here in its canonical portfolio reading (anchor = `R_f`, increment = the
excess return) — is the streaming portfolio primitive (`O(n_w)`, no share-axis): it solves the
continuous share choice per cell (scan + Newton), seats the per-cell optimal `θ*(x)`, and pushes mass
through the chosen truncated-Gaussian return row. The excess-return moments `(μ_x, σ_x)` are
moment-matched in `model.jl` to the calibration's two-point lottery `(R_risky, p_risky)`. Higher `θ`
raises both the mean and variance of next wealth, so a risk-averse CRRA agent picks an **interior**
share — exactly the mean–variance tradeoff.

## Equilibrium

Returns are **exogenous** (partial equilibrium), so there is no market to clear: the outer loop is a
single `solve_steady_state_given_env!`. The borrowing constraint (`b' ≥ 0`, the grid floor) plus
impatience (`β·E[R] < 1`) give a stationary wealth distribution.

## Run

```julia
julia --project=. examples/portfolio/steady_state.jl
```

Reports mass conservation, mean wealth, and the risky-share policy range. With the default
calibration (`σ = 3`, 3% premium) the poorest, borrowing-constrained agents go all-risky (little to
lose) while the wealthy hold a smaller interior share (≈ 0.6 at the top of the distribution) — the
standard risk-aversion gradient. (The two grid endpoints read `θ* = 0` by convention/artifact: the
`w = 0` cell has nothing at stake, and at `w_max` the clamped up-landing leaves risk with no upside.)
