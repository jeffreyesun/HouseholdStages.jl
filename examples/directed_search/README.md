# Directed job search (Moen 1997; Menzio–Shi 2011)

An income-fluctuation savings problem with a **directed-search** block: an
unemployed worker chooses *which posted-wage submarket to apply to*, trading off
a higher wage against a lower job-fill probability. The point of this example —
like `examples/portfolio` — is that the entire household block is a composition
of **existing library stages, with no bespoke household stage rolled here**.

## Household block

Time order (`s1 ∘ s2` runs `s1` first):

`DirectedSearch ∘ Matching ∘ Receipt ∘ ConsumptionSavings`

| Stage | Library stage | What it does |
|---|---|---|
| `DirectedSearch` | `DirectedSearchStage` (= `LogitChoiceStage` over `:submarket`) | The unemployed pick a posted-wage submarket (logit, dispersion `ε`). The search cost varies along `:employment`: the **employed pay `+Inf` to switch** (locked to their job's wage), the unemployed aim freely. |
| `Matching` | `MarkovStage` on `:employment`, transition varying along `:submarket` | Unemployed → employed at wage `w_j` w.p. the fill probability `f(w_j)`; employed → unemployed w.p. the separation rate `s`. |
| `Receipt` | `WealthChangeStage` | Cash-on-hand `a ↦ (1+r)·a + income`; `income = w_j` if employed in submarket `j`, the benefit `b` if unemployed. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` | Picks next-period wealth on the wealth grid; `c = x − a'`. |

**Where the wage/fill-probability tradeoff lives.** It is *not* a kwarg of the
directed-search stage. Because `DirectedSearch` runs **first** and `Matching`
**second**, the logit's continuation value `V_end[submarket]` — the value of being
in submarket `j` going into the matching draw — already integrates "match w.p.
`f(w_j)` into a job paying `w_j` vs. stay unemployed." That is exactly the
Menzio–Shi submarket-value channel: the tradeoff enters through the continuation
value set by the following stages, and the directed-search stage is a plain logit
over the submarket axis.

The fill schedule `f(w)` is a clamped **decreasing** linear map from `f_hi` (at
the lowest posted wage) to `f_lo` (at the highest) — high posted wage ⇒ tight
market ⇒ low fill probability.

## State axes

- `:wealth` — log-spaced grid, `[0, w_max]`, `N_w` points.
- `:employment` — `[:unemp, :emp]`.
- `:submarket` — the posted wages `p.wages` (the submarket index *is* its wage).

## Equilibrium

Partial equilibrium: the return `r`, the benefit `b`, and the fill schedule
`f(·)` are exogenous (the Moen/Menzio–Shi free-entry tightness schedule is
summarised by `f(w)`). There is no market to clear, so the outer loop is a single
`solve_steady_state_given_env!`. Impatience (`β(1+r) < 1`) plus the wealth floor
deliver a stationary distribution.

## Running

```julia
julia --project=. examples/directed_search/steady_state.jl
```

At the default calibration (`N_w = 150`, 7 submarkets over posted wages
`[0.8, … , 1.4]`, `ε = 0.05`, `s = 0.05`) the steady state has an unemployment
rate ≈ 11%, mean wealth ≈ 1.8, and an **interior** search choice: the unemployed
concentrate on intermediate-to-high submarkets (wages 1.2–1.3, fill probs
0.48–0.37) rather than the maximum-wage one (1.4, fill prob 0.25) — the directed-
search tradeoff is operative.

The regression test is `test/test_example_directed_search.jl`.
