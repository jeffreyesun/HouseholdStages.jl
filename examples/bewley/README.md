# Bewley — standard incomplete-markets self-insurance

The canonical Bewley / standard-incomplete-markets model (Bewley
1977/1986; the imperfect-insurance tradition of Aiyagari 1994 and Huggett
1993). A household faces idiosyncratic income risk with access to only a
single risk-free asset, so it cannot insure the income shock directly — it
**self-insures** by accumulating a precautionary buffer stock of wealth.

This example is a *partial-equilibrium* experiment: the interest rate `r`
is fixed and exogenous, and a single steady-state solve reads off the
precautionary wealth distribution that idiosyncratic risk generates at that
`r`. There is no market to clear.

## The household block

The within-period problem is the canonical L03 / L04 three-stage chain —
**the same chain as Aiyagari and Huggett, which is expected**: the value of
the example is the Bewley *framing* (fixed-`r` self-insurance), not a new
stage.

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

| Chain role | Library stage | What it does |
|---|---|---|
| `IncomeShock` | `MarkovStage` (`axis = :income`) | Idiosyncratic income Markov draw; K-operator is `P_y`. |
| `IncomeReceipt` | `WealthChangeStage` (`axis = :wealth`) | Deterministic receipt `a ↦ (1+r) a + y`: gross interest plus this period's endowment. Backward interpolates `V_end` linearly along the asset axis; forward redistributes mass to the asset grid. |
| `ConsumptionSavingsStage` | `ConsumptionSavingsStage` (`axis = :wealth`) | Choose next-period assets `a_end` on the asset grid; implicit budget `c = a_in − a_end`; CRRA utility. Inner argmax via divide-and-conquer monotone search. |

No bespoke household stage: the block is three existing exported stages
composed with `∘`. Two precautionary moments are attached via
`define_moments!`:

- `A_mean = ∫ a dΛ` — the aggregate self-insurance buffer stock.
- `frac_constrained = ∫ 𝟙{a ≈ a_min} dΛ` — the hand-to-mouth share pinned
  at the borrowing constraint.

## Equilibrium notes

The defining feature is the **fixed, exogenous** interest rate, strictly
below the impatience knife-edge:

```
r  <  1/β − 1
```

- `r < 1/β − 1` makes `β(1+r) < 1`: the household is impatient enough that
  it runs wealth *down* in good income states, so the **precautionary
  motive** — not a drift toward the grid top — pins a non-degenerate
  ergodic wealth distribution. This is the standard-incomplete-markets
  stationarity condition.
- At `r = 1/β − 1` exactly, the Chamberlain–Wilson result says wealth
  diverges (perfect self-insurance in the limit). The `gap = 1/β − 1 − r`
  reported by the driver is positive by construction.
- Because returns are exogenous there is **no market-clearing outer loop**.
  The whole solve is a single `solve_steady_state_given_env!` at the fixed
  `r` — contrast Aiyagari (tatonnement on `K`) and Huggett (bisection on
  `r` to clear the bond to zero). To turn this into a closed Huggett/
  Aiyagari general equilibrium you would wrap an outer loop that drives a
  market residual; that machinery deliberately lives with the consumer, not
  in the library.

The asset grid is log-spaced on `[a_min, a_max]`: dense near the borrowing
constraint (where the buffer-stock policy is most nonlinear) and coarse at
the top. `WealthChangeStage.backward` interpolates `V_end` linearly, so the
grid top must be far enough out that `(1+r) a + y` stays inside the grid for
active cells — the standard Aiyagari log-grid argument.

## Calibration and result

Default `BewleyParams`: β = 0.95, σ = 2.0, **r = 0.03 fixed** (vs.
`1/β − 1 = 0.0526`, an impatience gap of 0.0226); three-state persistent
income with `y_grid = [0.5, 1.0, 1.5]`; `N_a = 400` log-spaced asset points
on `[0, 120]` with a zero-borrowing constraint.

```
r                  = 0.0300   (1/β − 1 = 0.0526, gap = 0.0226)
ΣΛ                 = 1.000000
A_mean (buffer)    = 2.4995
frac at constraint = 0.0376
```

The strictly positive `A_mean` is the self-insurance buffer stock that
idiosyncratic income risk generates at this fixed return; ~3.8% of
households sit at the borrowing constraint (the hand-to-mouth share).
Lowering `r` (a larger impatience gap) shrinks the buffer and pushes more
mass to the constraint; raising `r` toward `1/β − 1` inflates it. Raising σ
(more risk aversion) strengthens the precautionary motive and the buffer.

## Relation to the sibling examples

The household block is byte-near `examples/aiyagari` and `examples/huggett`
— that redundancy is expected and intentional (each example owns its own
model primitives). What is distinct here:

- **vs. Aiyagari** — no Cobb-Douglas production and no tatonnement on `K`;
  `r` is fixed exogenously rather than `= F_K − δ`. Partial equilibrium.
- **vs. Huggett** — no bond-market clearing (`∫ a dΛ → 0`) and no bisection
  on `r`; here `r` is a fixed parameter and `∫ a dΛ` is a *reported
  outcome*, not a residual driven to zero. Bewley keeps a zero-borrowing
  constraint and a single fixed return to isolate the self-insurance
  mechanism.

## Run

```bash
julia --project=. examples/bewley/steady_state.jl
```

A single fixed-`r` solve; about 6 seconds after first compilation at
`N_a = 400`.

## Files

- `model.jl` — parameters, layout, the three-stage chain, CRRA utility,
  `bewley_env`, and the precautionary moments.
- `steady_state.jl` — the single fixed-`r` `solve_steady_state_given_env!`
  driver and its report.
- `../../test/test_example_bewley.jl` — module-wrapped regression test.
