# Fagereng–Guiso–Malacrino–Pistaferri (2020) — return heterogeneity

Persistent **heterogeneity in returns to wealth** (Fagereng, Guiso, Malacrino & Pistaferri,
*Heterogeneity and Persistence in Returns to Wealth*, ECMA 2020). Households differ persistently in
the gross return they earn on wealth; a high-return type compounds faster, so the right tail of the
wealth distribution thickens far beyond what income risk alone produces. The within-period problem
is **five existing `HouseholdStages` stages, no bespoke household stage** — the demonstration this
example exists for (Part 3: literature household blocks expressible from the library alone).

## Household block

In time order:

```
ReturnType ∘ IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ ReturnReceipt
```

| stage | library stage | does |
|---|---|---|
| `ReturnType` | `MarkovStage` | persistent return type on `:rtype` |
| `IncomeShock` | `MarkovStage` | idiosyncratic labor income on `:income` |
| `Receipt` | `WealthChangeStage` | `a ↦ a + w·y` (cash-on-hand) |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | pick saved wealth `b'`, `c = x − b'`, CRRA |
| `ReturnReceipt` | `WealthChangeStage` | `b' ↦ R(rtype)·b'` — the **type-specific gross return** |

## The key design decision

`GaussianLoadingStage` — the natural portfolio primitive — takes **plain** return moments
(`anchor`/`increment_mean`/`increment_sd`, scalars or `FromEnv`); those are fixed across cells, so it *cannot* vary the mean
return by a persistent `:rtype` axis. Fagereng's contribution is precisely heterogeneity in the
**mean** return across people, so the return must read the type. A `WealthChangeStage` does exactly
that: its `wealth_post` closure reads any layout axis, so

```julia
wealth_post = (; rtype, wealth, env) -> env.R_by_type[Int(rtype)] * wealth
```

carries the per-type return cleanly and faithfully — "the return process is the model's, not the
package's" (catalog §2). `R(rtype)` is supplied via `env`, so the heterogeneous and homogeneous
calibrations are a pure `env` swap on an identical block.

A homogeneous `GaussianLoadingStage` portfolio leg composes cleanly on top (see `examples/portfolio`),
but it would add an idiosyncratic-risk margin that is *not* the Fagereng mechanism, so it is omitted.

## Equilibrium

Returns, the type process, income, and the wage are **exogenous** (partial equilibrium): the outer
loop is a single `solve_steady_state_given_env!`. Impatience relative to every type's return
(`β·R(rtype) < 1` for all types) plus the borrowing constraint (`b' ≥ 0`) give a stationary
distribution.

## Run

```julia
julia --project=. examples/fagereng/steady_state.jl
```

Solves the block **twice** on the same chain — heterogeneous returns `R = [1.00, 1.04]` vs
homogeneous `R = [1.02, 1.02]` (same population-average return) — and shows that return
heterogeneity **fattens** the wealth distribution. With the default calibration:

| | heterogeneous | homogeneous |
|---|---|---|
| CV of wealth | **2.16** | 0.69 |
| top-10% share | **0.33** | 0.23 |

The high-return type holds ~1.6× the wealth of the low-return type; under homogeneous returns the
two types are identical. The dispersion gap (CV ~3× larger, top-10% share ~9.5 pp larger) is the
fat-tail signature of persistent return heterogeneity.
