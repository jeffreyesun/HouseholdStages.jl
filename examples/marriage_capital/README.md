# Marriage-specific capital (Becker 1973) — within-match life cycle

A couple invests in match-specific capital `m` that raises the surplus of staying
together. The capital decays, and the per-period **separation hazard falls** with
match capital: better-invested matches are more stable.

## Household block — existing stages only

```
Household block = MatchInvestStage ∘ SeparationStage
                = CapitalInvestmentStage(:match_capital)  ∘  MarkovStage(:married | match_capital)
```

State = `(match_capital, married)`, with `:married` the 2-state axis `[1, 0]`
(married, separated/absorbing). `∘` runs the LEFT stage first: invest in the match,
then draw the separation shock on the chosen match capital.

- **`CapitalInvestmentStage(:match_capital)`** — from match capital `m` the couple chooses
  next stock `m'`, paying a convex investment cost
  `effort_cost(i; env) = R·(i/a)^{1/γ}` on gross investment `i = m' − (1−δ)m`
  (`δ` = match depreciation) and earning the match flow `production(m) = R·m`
  (Becker's match-specific surplus). Identical stage to `examples/human_capital`
  and `examples/health`, different flow.
- **`MarkovStage(:married | match_capital)`** — a 2×2 row-stochastic transition
  with the separated state absorbing, supplied as a match-capital-dependent dep
  closure `(; match_capital) -> [stay(m) 1−stay(m); 0 1]`. The retention
  probability `stay(m)` is a **logistic increasing in `m`**, so the separation
  hazard `1−stay(m)` falls with match capital. Economic data fed to the existing
  stage — not a new stage.

This is **structurally identical to `examples/health`** (Grossman): a Ben-Porath
investment in a stock composed with a stock-dependent survival Markov. The relabel
is `:health→:match_capital`, `:alive→:married`, mortality→separation hazard.

## Driver (example-side, allowed)

`steady_state.jl` is a finite-horizon **within-match cohort** solve, exactly like
`examples/health/steady_state.jl`: backward induction over durations `N_age…1`
seating each period's investment policy, then a forward sweep of a couple "born"
married at `m0`. Each period a `1−stay(m)` share of the married mass moves into the
absorbing separated state, so the married sub-mass traces the separation curve. The
duration-specific investment efficiency `a` is threaded through `env` by the driver.

### Why finite-horizon, not stationary

A stationary distribution with separation but no match **formation** leaks all mass
into the absorbing separated state. A nondegenerate stationary married-mass would
need a forward mass-injection source (new couples entering), i.e. the two-sided
matching / formation problem. The within-match cohort sidesteps that honestly.

## Catalog status: within-match ✅ (formation = matching gap)

This example builds **only the within-match investment + capital-dependent
separation hazard**. The two-sided match **formation / bargaining** (who marries
whom, how surplus is split, the inflow of new couples) is **out of scope** — a
matching gap, not a household-stage composition. What is built here is the Becker
within-match margin in full.

## Run

```
julia --startup-file=no --project=. examples/marriage_capital/steady_state.jl
```

Confirmed output: total grid mass conserved at `1.000000` every duration (mass
accumulates in the absorbing separated state), `V` finite everywhere, married rate
decaying `1.00 → 0.67 (mid) → 0.02 (end)` along the separation curve, separation
hazard `0.76` at low `m` vs `0.01` at high `m` (falls with capital), mean match
capital among the still-married hump-shaped (peak `16.78` at duration 9), expected
relationship length `19.6` periods.
