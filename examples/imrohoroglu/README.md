# İmrohoroğlu (1989) — liquidity-constrained self-insurance

İmrohoroğlu's (1989) cost-of-business-cycles economy in steady-state,
partial-equilibrium form. A household faces a two-state **employment**
Markov process (employed / unemployed) and self-insures with a single
storage asset. The unemployment benefit is a low endowment in the
unemployed state, carried directly in the employment-axis grid
(`y_grid = [0.25, 1.0]`).

## The household block

The within-period problem is the canonical three-stage spine — the same
spine as Aiyagari / Bewley / Huggett, with the income shock relabelled as
an employment shock:

```
EmploymentShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

| Chain role | Library stage | What it does |
|---|---|---|
| `EmploymentShock` | `MarkovStage` (`axis = :employment`) | Employed/unemployed Markov draw; K-operator is `P_e`. |
| `IncomeReceipt` | `IncomeStage` (`axis = :wealth`) | Receipt `a ↦ (1+r) a + w·e`, where `e` is the employment-axis endowment value (1.0 / 0.25). |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` (`axis = :wealth`) | Choose next-period assets; implicit budget `c = a_in − a_end`; CRRA utility. |

No bespoke stage — three existing exported stages composed with `∘`,
parameterized by the transition matrix `P_e`, the log wealth grid, and the
receipt closure. Two precautionary moments are attached via
`define_moments!`: `A_mean` (aggregate buffer stock) and
`frac_constrained` (mass at the `a = 0` liquidity constraint).

## Equilibrium framing

The return `r` is **fixed and exogenous**, strictly below `1/β − 1`, so the
whole solve is a single `solve_steady_state_given_env!` — no market is
cleared. This is the pure partial-equilibrium self-insurance experiment.
What makes it İmrohoroğlu rather than Bewley is the two-state
employment/unemployment shock and the unemployment-benefit endowment.

## Result

Default `ImrohorogluParams`: β = 0.96, σ = 1.5, **r = 0.03 fixed** (vs.
`1/β − 1 = 0.0417`, gap 0.0117); `N_a = 250` log-spaced asset points on
`[0, 80]`.

```
r                  = 0.0300   (1/β − 1 = 0.0417, impatience gap = 0.0117)
ΣΛ                 = 1.000000
A_mean (buffer)    = 1.4386
frac at constraint = 0.0055
```

## Run

```bash
julia --project=. examples/imrohoroglu/steady_state.jl
```

A single fixed-`r` solve; about 7 seconds after first compilation.

## Files

- `model.jl` — parameters, layout, the three-stage chain, CRRA utility,
  `imrohoroglu_env`, and the precautionary moments.
- `steady_state.jl` — the single fixed-`r` solve driver and its report.
