# One-sided Partner Choice (Choo–Siow 2006)

Choo & Siow (2006, *JPE*) show that, under additively-separable type-I
extreme-value match utility, the equilibrium matching probabilities of a
marriage market **are a logit**: the log-odds that an agent matches a
partner of type `k` (versus staying single) equal the *systematic* match
payoff of that pairing. This example takes that insight literally and
builds the **one-sided** problem — the searching side — as a stationary
household block, with the other side's availability taken as exogenous and
supplied through `env`.

The whole household block is assembled from **existing library stages** —
no bespoke household stage, kernel, or per-cell value/transition logic is
defined in this example. The Choo–Siow logit *is* `LogitChoiceStage`; that
is the point of the example.

## Household block

A searcher carries a single discrete state — their marriage state:

```
marital ∈ {single, matched-to-type-1, …, matched-to-type-K}
```

Within-period decomposition, in time order:

```
Flow ∘ Discount ∘ PartnerChoice ∘ Dissolution
```

| Stage (this example) | Library stage | What it does |
|---|---|---|
| `Flow` | `UtilityStage` (array form) | Per-period flow payoff by marital state: the single's outside flow `u_single`, a match-to-`k`'s companionship flow `α[k]`. State-only `V`-shifter. |
| `Discount` | `TimeDiscountingStage` (`β`) | `V_start = β · V_end`. Supplies the contraction that makes the stationary value recursion well-posed. |
| `PartnerChoice` | `LogitChoiceStage` (axis `:marital`, `ε`) | **The Choo–Siow logit.** Singles draw a Gumbel and pick a destination — stay single, or match type `k` with systematic payoff `Π[k]` entered as a negative transition cost `C[single, matched-k] = −Π[k]`. Already-matched agents are pinned (cost `+Inf` to every destination but their own) and do not re-choose. |
| `Dissolution` | `MarkovStage` (axis `:marital`) | Each match dissolves to `single` at exogenous hazard `δ`; singles stay single. Row-stochastic `T[from, to]` built by `dissolution_matrix`. |

State space: `(marital,)` — a single categorical axis with `K+1` values.

The cost matrix `C[origin, dest]` is the only place the model's economics
live, and it is plain data built from primitives by `partner_cost_matrix`
(an outer-loop helper, not a stage), handed to the stage via
`FromEnv(:cost_matrix)`. The systematic payoff reflects the other side:

```
Π[k] = a[k] + ω · log(n[k])
```

with `a[k]` intrinsic attractiveness and `n[k]` the exogenous availability
of partner type `k` (read from `env`). `ω = 1` recovers the canonical
Choo–Siow form in which log-availability enters the match log-odds
one-for-one. At `ε = 1`, the single state's choice probabilities satisfy
`log(π_k / π_0) = Π[k] + (V_matched-k − V_single)` — the Choo–Siow matching
identity, with the continuation-value term reflecting that this is a
*dynamic* (recurring) market rather than a one-shot assignment.

The single↔matched flow — singles match in via the logit; matches dissolve
back via the hazard — makes the marital chain ergodic, so
`solve_steady_state_given_env!` returns a non-degenerate stationary
distribution over marriage states: the cross-section of a stationary
marriage market.

## Equilibrium notes

This is the **one-sided** block: the partner side is exogenous, so a single
inner V/Λ fixed-point solve at the env *is* the steady state — there is no
outer loop in `steady_state.jl`.

The **two-sided** Choo–Siow equilibrium — where both sides' availabilities
clear each partner type's matching market and the `n[k]` are endogenous — is
a known outer-loop gap. It would add a fixed point on the availabilities
`n[k]` layered *outside* this household block, exactly as Aiyagari's
tatonnement on `K` wraps its inner solve. Nothing in the household block
changes; only the driver would close it. Solving the two-sided assignment
in one shot is *not* a recurring household decision and would need bespoke
matching logic, so it is deliberately out of scope for a "library stages
only" household block.

Headline result at the default calibration
(`β = 0.94`, `δ = 0.10`, `ε = 1.0`, `a = [0.8, 0.4, 0.0]`,
`α = [0.6, 0.5, 0.3]`, symmetric availability `n = [1, 1, 1]`):

```
single share = 0.167   match rate = 0.833
stationary marital distribution:
  single        Λ = 0.1667
  matched_k1    Λ = 0.5741   (Π = +0.800)
  matched_k2    Λ = 0.2145   (Π = +0.400)
  matched_k3    Λ = 0.0447   (Π = +0.000)
```

The most attractive partner type (`k1`, highest `Π`) holds the most matched
mass, the least attractive (`k3`) the least — the logit sort over partner
types. Raising one type's availability `n[k]` pulls stationary mass toward
matches with that type (the Choo–Siow comparative static); raising the
dissolution hazard `δ` raises the single share.

## How to run

```bash
# Stationary solve + marital-distribution report
julia --project=. examples/choo_siow/steady_state.jl

# Test (asserts mass(Λ) ≈ 1, finite V, ergodic non-degenerate distribution,
#       δ comparative static, Choo–Siow availability comparative static)
julia --project=. -e 'using HouseholdStages; include("test/test_example_choo_siow.jl")'
```
