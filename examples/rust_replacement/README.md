# Rust (1987) optimal engine replacement

Harold Zurcher's bus-engine problem (Rust, *Econometrica* 1987) — the canonical
**regenerative optimal-stopping** model. The single state is engine **mileage**
`x` on a grid; each period the manager chooses `d ∈ {keep, replace}`:

- **KEEP** — pay the operating cost `c(x)` (rising in mileage); mileage then
  deteriorates by a stochastic increment.
- **REPLACE** — pay the replacement cost `RC` plus the operating cost of a fresh
  engine `c(0)`; mileage **resets to 0**, then deteriorates from 0.

The binary choice is smoothed with i.i.d. Gumbel taste shocks of scale `ε` (the
EV-logit). There is **no consumption/savings core** — "value" is just the
discounted stream of negative costs. The defining feature, versus the existing
`technology_adoption` (a network-externality adopt logit with no regeneration),
is the **reset-to-zero** that makes the problem regenerative — and it falls
straight out of composition, as a following `WealthChangeStage`.

The household block is **existing library stages only — no bespoke stage.**

## Household block

Within-period decomposition, in time order (forward order, left → right):

```
Advance ∘ Choose ∘ FlowCost ∘ Discount ∘ Reset ∘ Forget
```

| Stage | Library stage | What it does |
|---|---|---|
| `Advance` | `MarkovStage(:mileage)` | Stochastic deterioration increment. **Leads** the block so the chain's boundary layout carries `:decision` at size 1. |
| `Choose` | `LogitChoiceStage(:decision ∈ {keep, replace})` | EV-logit keep/replace, growing the transient `:decision` axis 1 → 2. Cost matrix is all-zeros — every payoff is a *destination* payoff, hence V-additive, hence in `FlowCost`. |
| `FlowCost` | `UtilityStage` | Contemporaneous payoff: `−c(x)` on keep, `−(RC + c(0))` on replace. |
| `Discount` | `TimeDiscountingStage(β)` | β on the continuation only (the flow cost stays undiscounted). |
| `Reset` | `WealthChangeStage(:mileage)` | The regeneration: replace → 0, keep → x. Reads `:decision`. Its output is next period's boundary mileage. |
| `Forget` | `ForgetfulSumStage(:decision)` | Collapse the transient decision axis 2 → 1. |

Writing the period boundary **before** the mileage shock (Rust's expected-value
form `EV`), the backward sweep reproduces exactly

```
EV(x⁻) = E[ W(x) | deteriorate from x⁻ ],
  W(x)         = ε·log[ exp(v(x,keep)/ε) + exp(v(x,replace)/ε) ],
  v(x,keep)    = −c(x)        + β·EV(x),
  v(x,replace) = −(RC + c(0)) + β·EV(0).
```

`x⁻` is the mileage entering the period; `x` the mileage after deterioration
(observed before the choice). Forward, deterioration moves each engine's mileage
up, the logit splits that mass across the two decisions by the softmax, `Reset`
sends the replace branch to 0, and `Forget` sums the branches — **mass is
conserved** (no exit).

### Why `Advance` leads (a framework detail)

The growing logit **cannot be the first stage**: a `ChainStage` inherits its
boundary (input) layout from the first stage's *construction* layout, and the
logit is necessarily built on the full destination size 2 (`:decision ∈ {keep,
replace}`). Leading with `Advance` (built on the decision-singleton layout) keeps
the period boundary at size 1, exactly as the exit composite keeps its growing
`:exiting` choice *inside* the chain. The reformulation is itself canonical — it
is Rust's conditional-expectation (`EV`) form, with the manager acting on the
realized post-deterioration mileage.

## Equilibrium notes

Costs are exogenous (partial equilibrium): no market to clear, so the outer loop
is a single `solve_steady_state_given_env!`. The `:decision` axis is transient
and gone by block end, so the stationary `Λ` is a pure mileage distribution; the
driver reconstructs the per-mileage replacement probability `π(replace | x)` from
the converged `(V, Λ)` with the same softmax (plain driver arithmetic, no block
internals) and weights it by the choice-time (post-deterioration) distribution
`Λ_choice = Tᵀ·Λ`.

Headline result (`β = 0.90`, `ε = 1.0`, `RC = 150`, `c(x) = x`, grid `[0, 30]`):

```
mean operating mileage  = 11.10
replacement rate        = 0.0336 per period   (≈ every 30 periods)
π(replace) crosses ½ at mileage = 22.0         (73% of the grid)
π(replace) at x=0 / mid / top   = 0.00 / 0.00 / 1.00
```

The textbook stopping outcome: replacement is **concentrated at high mileage**
(near-zero probability through the lower two-thirds of the grid, rising sharply
to certainty at the top), and no mass piles at the absorbing grid top — engines
are replaced before they reach it.

## How to run

```bash
julia --project=. examples/rust_replacement/steady_state.jl
```
