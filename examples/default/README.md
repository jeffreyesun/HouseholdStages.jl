# Sovereign / consumer default — incomplete markets with a repay/default choice

A borrower that each period chooses **repay vs default** on its outstanding debt (Eaton–Gersovitz
1981; Arellano 2008; Chatterjee–Corbae–Nakajima–Ríos-Rull 2007). The within-period problem is
**five existing `HouseholdStages` stages, no bespoke household stage** — the demonstration this
example exists for (Part 3: literature household blocks expressible from the library alone). The
repay/default branches are wired purely by composition: the `DefaultStage` picks the status, and the
state consequences of default (debt discharge, income haircut, the exclusion spell) are *following*
stages that read `cell.status` — the same pattern as `BuyHomeStage ∘ WealthChangeStage`.

## Household block

In time order:

```
IncomeShock ∘ DefaultChoice ∘ DebtReset ∘ Receipt ∘ Savings ∘ Readmission
```

| stage | library stage | does |
|---|---|---|
| `IncomeShock` | `MarkovStage` | draw next endowment on the `:income` axis |
| `DefaultChoice` | `DefaultStage` | gated argmax on a 2-level `:status` axis: good-standing agents pick repay (stay 1, score 0) or default (→ 2, score `−χ`); excluded agents (`avail`-gated) stay excluded |
| `DebtReset` | `WealthChangeStage` | `cell.status == 2 ? 0 : a` — an excluded/defaulting agent's debt is discharged |
| `Receipt` | `WealthChangeStage` | cash-on-hand `x = (1+r)·a + y`; an excluded agent suffers the income haircut `(1−λ)·y` (the Arellano output cost) and carries no debt |
| `Savings` | `ConsumptionSavingsStage` | pick next assets `a' ∈ grid` (floor `a_min < 0` is the borrowing limit), `c = x − a'`, CRRA utility |
| `Readmission` | `MarkovStage` | an excluded agent regains good standing next period w.p. `ψ`; `T = [1 0; ψ 1−ψ]` |

The continuation value `V_end` at each `:status` level — the repay value vs the default value — is
assembled by the four stages *after* `DefaultChoice`, so the `(max, +)` contraction in `DefaultStage`
is exactly the Eaton–Gersovitz comparison `max(V_repay, −χ + V_default)`. No bespoke per-cell value
logic: the branches are ordinary composition over `cell.status`-reading wealth closures.

## Why the persistent exclusion spell

Without a *forward-looking* cost of default, default is a costless per-period arbitrage — discharge
the debt, re-borrow to the floor, repeat — and the equilibrium degenerates to a 100% default rate.
The `Readmission` `MarkovStage` is the canonical fix and is itself a library stage: a defaulter is
excluded (income-haircut) for a geometric spell of mean `1/ψ ≈ 6.7` periods. That persistent cost
deters always-default and delivers a **non-degenerate** default rate driven by income risk, exactly
as the literature intends. (The model is stylized in one respect: exclusion is an income haircut, not
full autarky — a status-dependent *choice* constraint `a' ≥ 0` for excluded agents is not expressible
through `ConsumptionSavingsStage`'s consumption-only utility closure, so the calibration instead makes
excluded agents endogenously choose not to re-lever.)

## Equilibrium

Prices are **exogenous** (a risk-free unit bond, partial equilibrium), so there is no market to
clear: the outer loop is a single `solve_steady_state_given_env!`. With the default calibration
(`β = 0.92`, `σ = 2`, haircut `λ = 0.40`, readmission `ψ = 0.15`, borrowing limit `a_min = −0.35`) the
stationary distribution has **mean assets ≈ −0.21** (net debtors), an **excluded mass ≈ 0.39**, and a
two-sided default policy (some good-standing cells repay, some default). The implied per-period
default rate among good-standing agents is `ψ · excluded / good ≈ 10%`.

## Run

```julia
julia --project=. examples/default/steady_state.jl
```

Reports mass conservation, mean assets, the excluded rate, and the inner V/Λ iteration counts.
