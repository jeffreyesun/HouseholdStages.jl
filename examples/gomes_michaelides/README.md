# Gomes–Michaelides (2005) — participation cost + preference heterogeneity

Limited stock-market **participation** the Gomes–Michaelides way: a fixed participation cost plus a
spread of **preference types**. Two margins coexist — an **extensive** margin (pay the fixed cost and
hold equity, or stay out) and an **intensive** margin (the risky share, conditional on entry). The
fixed cost sorts households into participants and non-participants; the preference spread lets the
model speak to both the participation rate and conditional shares at once. The within-period problem
is the **`⊕`-over-preference-type direct sum of the Merton/portfolio chain — no bespoke household
stage** (the demonstration this example exists for).

## Household block

Per preference type `i`, in time order, then summed over types:

```
block_i   = IncomeShock ∘ Receipt ∘ ConsumptionSavings(σ_i) ∘ Portfolio(F)
household = product(block_1, …, block_n; axis = :ptype)
```

| stage | library stage | does |
|---|---|---|
| `IncomeShock` | `MarkovStage` | draw next income on `:income` |
| `Receipt` | `WealthChangeStage` | `a ↦ a + w·y` (cash-on-hand) |
| `ConsumptionSavings(σ_i)` | `ConsumptionSavingsStage` | pick `b'`, `c = x − b'`; CRRA with **per-type curvature `σ_i`** captured in the utility closure |
| `Portfolio(F)` | `MeanVarianceStage` | pick risky share `θ`; **fixed participation cost** via `cost = (θ; env) -> θ > 0 ? env.F : 0.0` |

The extensive margin **is** the `MeanVarianceStage` `cost` closure: staying out (`θ = 0`) is free, any
positive share pays `F`, so the household participates only when an optimal positive share clears the
fixed cost. The `product` direct sum is block-diagonal (a household keeps its type forever), so the
**standard solver runs directly** — each type's slice converges independently.

## Distinct-σ `product` — it works

The headline technical question: can `product` glue blocks that differ in CRRA `σ`? **Yes.** `σ` rides
the `ConsumptionSavingsStage` utility **closure** `(cell, c; env) -> u_crra(c, Val(σ))`, which captures
`σ::Float64`. Two such closures have the **same concrete type** (only the captured value differs), so
the per-type chain Spec types are identical — exactly the uniformity `product` asserts
(`typeof(component) === first_type`). No fallback to β-heterogeneity was needed; the blocks differ
only in the captured `σ` and `product` accepts them cleanly. (Had `σ` been a *type parameter* rather
than a captured value, the Spec types would differ and `product` would reject them — the closure
capture is what keeps the type uniform.)

## Fidelity note — the cost is in value units

`MeanVarianceStage`'s backward computes `V_start(b') = max_θ[ Σ_k w_k·V_end(b'·R_{θ,k}) − cost(θ) ]`,
so `F` is an additive penalty in **continuation value (utils)**, not a wealth/consumption subtraction.
The library `cost` closure sees only `(θ; env)` — **never the cell's wealth** — so a wealth-denominated
fixed cost (the literal Gomes–Michaelides *monetary* entry cost) is **not expressible** through this
primitive. The faithful statement here is a flat per-period **utility** cost of entry.

This has a calibration consequence. With a flat utility cost the participation–wealth gradient **flips
sign at `σ = 1`**: the utils gain from the premium scales as `b'^{1-σ}`, which **falls** in wealth for
`σ > 1` (the rich would drop out) and **rises** in wealth for `σ < 1` (the textbook "rich participate"
gradient). We therefore calibrate the preference types with **`σ < 1`** (`σ ∈ {0.50, 0.85}`), so
participation rises with wealth as in the data. A `σ > 1` calibration with this same flat utility cost
would invert the gradient — an artifact of the cost being denominated in utils rather than goods, not
of the participation mechanism itself.

## What it produces

Partial equilibrium (exogenous returns, wage, cost), so a single `solve_steady_state_given_env!`.
With the default calibration (`β = 0.955`, 3% premium, `F = 0.025` utils):

- **Participation is partial and differs by type:** overall ≈ 0.34; by type ≈ 0.24 (`σ = 0.50`) and
  ≈ 0.44 (`σ = 0.85`). The more risk-averse `σ < 1` type accumulates more (stronger prudence),
  reaches the wealth where the fixed cost pays off more often, and participates more — the
  wealth-accumulation channel dominates the desired-share channel here.
- **Participation rises with wealth:** across equal-mass wealth terciles, ≈ `0.00 → 0.03 → 1.00`. The
  fixed cost creates a wealth threshold below which no one pays to enter — exactly the G–M extensive
  margin.
- **Conditional shares corner** (`θ̄ ≈ 1.0`): the risk-tolerant `σ < 1` types, tilted further toward
  equity by non-tradable labor income (the Cocco–Gomes human-wealth effect), pick the top of the
  share grid once in. The intensive margin is therefore degenerate under this calibration; the
  **extensive participation margin is the demonstrated contribution**.

## Run

```julia
julia --project=. examples/gomes_michaelides/steady_state.jl
```

Reports mass conservation, aggregate wealth, per-type participation and conditional share, and the
participation profile across wealth terciles.
