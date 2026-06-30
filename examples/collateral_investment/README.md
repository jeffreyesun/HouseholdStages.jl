# Collateral-constrained investment — Moll (2014) / Buera–Shin (2013) / Midrigan–Xu (2014)

Entrepreneurs operate capital subject to a **collateral constraint**: a firm with net
worth `a` can run capital only up to a multiple `λ` of its net worth, `k ≤ λ·a`. Financial
frictions therefore cause capital **misallocation** — a productive but poor entrepreneur
cannot reach its efficient scale — and the only escape is **self-financing** (slowly
accumulating net worth out of retained profit). The within-period firm block is built from
**existing library stages only**: it is the Aiyagari savings spine with operating profit
replacing labor income and the collateral cap folded into the profit closure.

## The dictionary: this *is* Aiyagari, relabeled

The whole model maps onto the Aiyagari incomplete-markets household by a change of names.
That mapping is the point of the example.

| Aiyagari household | Collateral-investment firm |
|---|---|
| wealth / assets `b` | **net worth `a`** (the operative savings axis) |
| labor income shock `y` | **idiosyncratic productivity `z`** |
| labor income `w·y` | **operating profit `π(a, z)`** |
| borrowing constraint `b ≥ b̲` | **collateral constraint `k ≤ λ·a`** |
| cash-on-hand `(1+r)b + w·y` | cash-on-hand `(1+r)a + π(a, z)` |
| save next assets `b'` | save next net worth `a'` (self-financing) |

The economically new object — the collateral constraint — does **not** introduce a new
state or a new stage. Because capital is **rented within the period** (see below), the
constraint is a static cap that lives entirely inside the profit function `π`, which sits
inside the cash-on-hand map of an ordinary `WealthChangeStage`.

## The chain (existing stages only)

State space: `(:wealth, :z)` — net worth `a` (log-spaced continuous grid) and idiosyncratic
productivity `z` (AR(1)-in-logs, Rouwenhorst-discretized to `N_z` states, normalized to
ergodic mean 1).

```
Produce ∘ SaveNetWorth ∘ ProductivityShock
= WealthChangeStage(cash = (1+r)a + π(a,z)) ∘ ConsumptionSavingsStage(:wealth) ∘ MarkovStage(:z)
```

| Stage (time order) | Library stage | What it does |
|---|---|---|
| `produce` | `WealthChangeStage(:wealth)` | Maps net worth `a` to cash-on-hand `(1+r)·a + π(a, z)`. The `wealth_post` closure reads **both** axes and `env` and computes `π(a,z)` with the `k = min(k*, λa)` collateral cap inline. **This closure carries the entire collateral mechanism.** |
| `savings` | `ConsumptionSavingsStage(:wealth)` | From cash-on-hand pick next net worth `a'`; implicit budget `c = cash − a'`; CRRA felicity. The self-financing margin. |
| `shock` | `MarkovStage(:z)` | Productivity transitions (persistent AR(1) in logs), drawn at period END for next period. |

Block: **`produce ∘ savings ∘ shock`** (`∘` runs the left stage first). Moments attached:
`mean_a`, `mean_k`, `mean_profit`, `mean_z`, `frac_constrained`, and the first two moments
of `log MPK` (for the misallocation dispersion).

## The design point: rented capital is a STATIC choice, so no new state and no new stage

In Moll (2014), capital is **rented period-by-period**, so capital is **not a state
variable** — it is a static within-period choice bounded by collateral, with a closed form.
The entrepreneur with net worth `a` and productivity `z` solves

```
π(a, z) = max_{k ≤ λ·a} [ z·k^α − (r+δ)·k ],
```

whose unconstrained optimum is `k* = (α z / (r+δ))^{1/(1−α)}`; the firm uses
`k = min(k*, λ·a)`. When `λ·a < k*` the **collateral constraint binds** (`k = λa`) — that is
the friction. Net worth then accumulates by ordinary saving out of cash-on-hand. So the
block is a clean composition of existing stages: a `WealthChangeStage` whose cash-on-hand
map embeds the constrained profit, then a `ConsumptionSavingsStage` on the net-worth axis,
then the productivity Markov chain.

**Contrast with capital as a carried state.** If capital were instead a *stock* the firm
carried across periods — `k` a second continuous axis, with a literal
`BorrowingConstraintStage` gate enforcing `k' ≤ λ·a'` — the constraint would couple two
continuous axes and require the §5(ii) **two-axis auxiliary-axis machinery** (a constraint
gate on one axis indexed by another). The static-rental form sidesteps all of that: the
constraint is a `min` inside a scalar profit function, and the block stays a one-dimensional
savings spine. The static-rental form is therefore both the **faithful** Moll decomposition
*and* the **clean** one — they coincide.

## Why `MarkovStage` is LAST here (Aiyagari puts it first)

With the shock last in time order, productivity `z` is realized at the start of each period
(carried in from the previous period's end-draw), production uses the contemporaneous
`(a, z)`, and — by stationarity — the end-of-chain distribution `Λ` is **exactly** the joint
of (net worth, current productivity) at the production point. So every production moment
(capital operated, fraction constrained, MPK) reads directly off `at_end`, with no need to
push `Λ` through the Markov kernel by hand. Backward induction is unaffected: the saver's
continuation is `E[V(a', z') | z]` either way.

## The misallocation mechanism

In the **frictionless** allocation (`λ → ∞`) every firm sets `k = k*`, so the marginal
product of capital `MPK = α z k^{α−1}` equals the common user cost `r + δ` for *all* firms —
MPK is **equalized**, capital is allocated efficiently. Under a binding constraint a firm
runs `k = λa < k*`, so its `MPK > r + δ`, and MPK **disperses** across firms. The
cross-sectional standard deviation of `log MPK` is the standard summary of misallocation
(zero in the frictionless benchmark); the fraction of firms with `λa < k*` measures how
widespread the friction is. Both are reported by the driver.

Prices (`r`) and the cross-sectional **wealth distribution** as an aggregate state are the
**caller's outer loop**. Here `r` is exogenous (partial equilibrium): no market to clear, so
the outer loop is a single `solve_steady_state_given_env!`. A general-equilibrium close would
clear the capital/bond market for `r` (and a wage, with a labor block) as in Aiyagari's
tatonnement, with the wealth distribution as the aggregate state.

## Calibration and result

Default `CollateralInvestmentParams`: `α = 0.33`, `δ = 0.06`, `r = 0.04`, `λ = 1.5`,
`β = 0.93` (so `β(1+r) < 1`, giving a stationary net-worth distribution), `σ = 1.5`;
7-state productivity with `ρ_z = 0.90`, `σ_z = 0.30`; `N_a = 600` log-spaced net-worth points
on `[0.05, 300]`.

```
V finite everywhere          = true   (range [-69.6, -7.1])
mass ΣΛ                       = 1.00000000
mean net worth  E[a]          = 11.89
mean productivity E[z]        = 1.00
mean profit  E[π]             = 1.40
mean capital operated E[k]    = 6.12
fraction constrained (λa<k*)  = 0.239
mean log MPK                  = -2.215   (frictionless: log(r+δ) = -2.303)
std  log MPK  (misallocation) = 0.213    (frictionless: 0)
k nondecreasing in a and in z = true
```

About 24% of firms are collateral-constrained — concentrated among low-net-worth,
high-productivity firms, exactly the entrepreneurs who *should* be operating at large scale.
Their elevated MPK pulls the mean above the frictionless `r+δ` and, more tellingly, produces
a `log MPK` dispersion of `0.21` where the frictionless allocation would deliver zero. The
`k(a, z)` table printed by the driver shows capital rising in net worth `a` until it hits the
unconstrained optimum `k*(z)` and then plateauing — the signature of self-financing relaxing
a binding constraint.

## Run

```bash
julia --project=. examples/collateral_investment/steady_state.jl
```

About 10 seconds at `N_a = 600`, `N_z = 7` after the first compilation pass.

## Files

- `model.jl` — parameters, productivity process, the closed-form collateral profit
  problem, layout, and the three-stage firm block.
- `steady_state.jl` — the single-solve partial-equilibrium driver and moment reporting.
