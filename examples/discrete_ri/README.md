# Discrete rational inattention — occupation choice (Matějka–McKay)

A stationary household that re-chooses a **discrete occupation** each period under
a Shannon information cost λ. The static rational-inattention discrete choice
*is* a generalized logit (Matějka & McKay, 2015), so the entire within-period
problem is **three existing library stages** — **no bespoke household stage is
rolled in this example**.

## Household block

Time order (forward): pick occupation under RI → income evolves given that
occupation.

| Stage (in time order) | Library stage | What it does |
|---|---|---|
| `OccupationChoice` | `LogitUtilityStage(axis = :occupation, ε = FromEnv(:λ), cost_matrix = C)` | RI choice over occupations: a temperature-λ logit (= `LogitChoiceStage ∘ UtilityStage`). The `UtilityStage` carries `u(z, a) = λ·log q(a) + flow(z, a)` — the Matějka–McKay **attention prior** `λ·log q(a)` (q read from `env.q`) plus the state-dependent wage payoff. The `LogitChoiceStage`'s `cost_matrix = C[a, a′]` is the origin-dependent occupation-**switching cost**. |
| `Discount` | `TimeDiscountingStage(β)` | `V_start = β·V_end`. |
| `IncomeDraw` | `MarkovStage(:income; transition = P_z(·\|occupation))` | The occupation just chosen sets next period's income process (occupations differ in income mean and persistence), so the discrete choice carries genuine **dynamic value**. The per-occupation transition is a row-stochastic matrix selected by the occupation axis — data handed to an existing stage. |

The whole block is `LogitUtilityStage ∘ TimeDiscountingStage ∘ MarkovStage`. No
`@definestage`, no kernel, no bespoke per-cell value/transition logic.

Because `ε = FromEnv(:λ)`, the **same** λ is both the logit temperature and the
weight on the log-prior, so the choice probabilities are exactly the
Matějka–McKay posterior `P(a′ | z, a) ∝ q(a′)·exp((flow(z, a′) + V(z, a′))/λ)`
(with the switching friction `exp(−C[a, a′]/λ)`).

## Equilibrium

Returns are exogenous (partial equilibrium): there is no price to clear. Two
ways to close the model, both rolled in `steady_state.jl`:

- **Fixed prior (headline).** Fix a uniform attention prior `q ∝ 1` and run a
  single inner V/Λ fixed-point solve (`solve_steady_state_given_env!`). This is
  the canonical RI reference point — the choice is the plain temperature-λ logit
  and the comparative static in λ is cleanest.
- **Endogenous prior (full Matějka–McKay).** Iterate `q` to the consistency
  condition `q(a) = ∫ 1{occ = a} dΛ` — the prior the strategy is optimized
  against must equal the realized choice share. A damped fixed point on `q`
  around the same inner solve. It seeks a corner when one occupation dominates
  (the known endogenous-prior degeneracy), so the fixed-prior solve is the
  headline.

**The comparative static (fixed uniform prior).** As the Shannon cost λ rises,
attention is costlier and the posterior collapses toward the uniform prior:
occupation shares compress toward 1/3 and the high-payoff "career" share falls
(0.82 at λ = 0.1 → 0.35 at λ = 8). Mean income falls with λ — the welfare cost
of inattention.

## Run

```bash
julia --project=. examples/discrete_ri/steady_state.jl   # solve + comparative static + endogenous-prior
julia --project=. examples/discrete_ri/model.jl          # just builds the household block
```

The regression test lives at `test/test_example_discrete_ri.jl`.

## Literature

Matějka & McKay (2015, AER), "Rational Inattention to Discrete Choices: A New
Foundation for the Multinomial Logit Model"; Steiner, Stewart & Matějka (2017,
ECMA) on the dynamic logit.
