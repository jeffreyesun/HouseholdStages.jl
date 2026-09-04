# Uninsured idiosyncratic investment risk — Angeletos (2007), Covas (2006)

Undiversifiable idiosyncratic **capital-return risk** adds a precautionary wedge to investment
(Angeletos, *Uninsured idiosyncratic investment risk and aggregate saving*, RED 2007; Covas,
*Uninsured idiosyncratic production risk with borrowing constraints*, JEDC 2006). An entrepreneur
invests in their own capital, whose return is risky and cannot be diversified away; a risk-averse
agent therefore tilts away from risky capital toward a safe store. The within-period problem is
**four existing `HouseholdStages` stages, no bespoke household stage** — the portfolio block
reinterpreted (Part 3: literature household blocks expressible from the library alone).

## Household block

In time order:

```
IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Investment
```

| stage | library stage | does |
|---|---|---|
| `IncomeShock` | `MarkovStage` | labor income on `:income` |
| `Receipt` | `WealthChangeStage` | `a ↦ a + w·y` (cash-on-hand) |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | pick saved wealth `b'`, `c = x − b'`, CRRA |
| `Investment` | `GaussianLoadingStage` (the portfolio stage read as a capital-exposure choice: anchor = `R_f`, increment = the capital excess return) | pick the continuous share `θ ∈ [0, 1]` of `b'` in **risky own capital**; next wealth `b'·(R_f + θ·(μ_x + σ_x·Z))` |

The "risky asset" is the agent's **own capital** with an undiversifiable idiosyncratic return — a
truncated-Gaussian excess with moments `(μ_x, σ_x)` matched in `model.jl` to the calibration's
mean-`μ_k`, half-spread-`Δ` two-point draw (`μ_x = μ_k − R_f`, `σ_x = Δ` at `p_up = ½`); the rest of
saved wealth earns the safe store `R_f`.
Raising θ raises both the mean and variance of next wealth, and a CRRA agent trades them off —
exactly `GaussianLoadingStage`'s mean–variance choice, now read as a capital-exposure choice. The
variance is genuinely idiosyncratic (each agent draws their own `R_k`), so the per-cell return
distribution is the right primitive — no aggregate risk, no diversification.

## Equilibrium

The capital-return distribution, the wage, and the safe return are **exogenous** (partial
equilibrium): the outer loop is a single `solve_steady_state_given_env!`. Impatience (`β·R_f < 1`)
plus the borrowing constraint (`b' ≥ 0`) give a stationary distribution.

## Run

```julia
julia --project=. examples/uninsured_investment_risk/steady_state.jl
```

Sweeps the idiosyncratic capital-return half-spread `Δ` (the variance) from a near-zero benchmark
upward, **holding the mean return `μ_k` fixed**, and shows the precautionary investment wedge. With
the default calibration (`σ = 3`, `μ_k = 1.08`, `R_f = 1.02`, 6% premium):

| `Δ` (variance ↑) | risky share θ* (grid) | mean risky capital | mean wealth |
|---|---|---|---|
| 0.005 (≈ riskless) | 0.98 | 66.5 | 68.5 |
| 0.20 | 0.90 | 18.1 | 30.3 |
| 0.40 | 0.49 | 2.6 | 6.3 |

As undiversifiable variance rises, the risky-capital share and the aggregate risky capital both
fall (mean risky capital drops ~96% end to end) — the precautionary wedge. The mean return is held
fixed throughout, so the decline is driven purely by the *variance* the agent cannot insure.
