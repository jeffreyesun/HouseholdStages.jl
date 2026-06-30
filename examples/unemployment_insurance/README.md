# Hansen–İmrohoroğlu (1992) — unemployment insurance

Hansen & İmrohoroğlu's (1992) economy with an explicit unemployment-
insurance scheme. As in İmrohoroğlu (1989), a household faces a two-state
**employment** Markov process and self-insures with a single storage asset.
The new ingredient is a genuine UI **policy**: the employed pay a flat
payroll tax `τ` and the unemployed receive a replacement rate `ρ` of the
wage. Net labour income is

```
employed:    w·(1 − τ)
unemployed:  ρ·w
```

carried in the **budget closure** — the employment axis is a 0/1 indicator,
and the closure maps it to net income via `τ` and `ρ` from `env`. This is
the distinction from `examples/imrohoroglu`, whose benefit lives directly
in the endowment grid.

## The household block

The within-period problem is the canonical three-stage spine:

```
EmploymentShock ∘ IncomeReceipt(UI) ∘ ConsumptionSavingsStage
```

| Chain role | Library stage | What it does |
|---|---|---|
| `EmploymentShock` | `MarkovStage` (`axis = :employment`) | Employed/unemployed Markov draw; K-operator is `P_e`. |
| `IncomeReceipt(UI)` | `IncomeStage` (`axis = :wealth`) | Receipt `a ↦ (1+r) a + [e·w(1−τ) + (1−e)·ρw]`, `e ∈ {0,1}`; `w, τ, ρ` from `env`. |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` (`axis = :wealth`) | Choose next-period assets; implicit budget `c = a_in − a_end`; CRRA utility. |

No bespoke stage — three existing exported stages composed with `∘`. The UI
policy enters **only** through the receipt closure (replacement rate and tax
in `env`), which is what distinguishes this example from the sibling
`imrohoroglu`. Two precautionary moments are attached: `A_mean` (buffer
stock) and `frac_constrained` (mass at the liquidity constraint).

## Equilibrium framing

The return `r` is **fixed and exogenous**, strictly below `1/β − 1`, so the
whole solve is a single `solve_steady_state_given_env!` — no market is
cleared (the UI budget is not closed here; `ρ, τ` are policy parameters).

## Result

Default `UIParams`: β = 0.96, σ = 1.5, **r = 0.03 fixed** (gap 0.0117);
UI policy **ρ = 0.25, τ = 0.03**; `N_a = 250` log-spaced asset points on
`[0, 80]`.

```
r                  = 0.0300   (1/β − 1 = 0.0417, impatience gap = 0.0117)
UI policy          : replacement ρ = 0.25, payroll tax τ = 0.03
ΣΛ                 = 1.000000
A_mean (buffer)    = 1.3643
frac at constraint = 0.0058
```

## Run

```bash
julia --project=. examples/unemployment_insurance/steady_state.jl
```

A single fixed-`r` solve; about 7 seconds after first compilation.

## Files

- `model.jl` — parameters (incl. UI `ρ, τ`), layout, the three-stage chain
  with the UI receipt closure, CRRA utility, `ui_env`, and the moments.
- `steady_state.jl` — the single fixed-`r` solve driver and its report.
