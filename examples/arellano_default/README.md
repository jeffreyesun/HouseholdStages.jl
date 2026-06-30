# Arellano (2008) — sovereign default with priced spreads

A small open economy with a stochastic endowment `y` issues one-period,
non-contingent **discount bonds** `a` (`a < 0` is debt). Each period in good
standing it chooses **repay** vs **default**. Defaulting discharges the debt but
triggers **exclusion** — autarky with the Arellano asymmetric output cost
`min(y, ŷ)` — from which the economy is readmitted with probability `ψ`.

What distinguishes this from `examples/default` (an *exogenous* risk-free bond)
is that the bond is priced by risk-neutral foreign lenders:

```
q(a', y) = (1/(1+rf)) · (1 − E[default(a', y') | y]),     budget:  c = x − q(a', y)·a'
```

The price depends on the **chosen** next assets `a'` **and** the current income
`y`. Default risk raises spreads on high debt and endogenously caps borrowing.
`q` is itself a fixed point: it depends on next period's default policy, which
depends on `q`.

## The dogfooding point

The entire within-period household block is a `∘` composition of **existing
exported stages** — no bespoke stage, no custom kernel:

```
IncomeShock ∘ DefaultChoice ∘ DebtReset ∘ Receipt ∘ Savings ∘ Readmission
   Markov       Default        Wealth-    Wealth-    Argmax       Markov
   (income)     (status)       Change     Change    (:wealth)    (status)
                               (:wealth)  (:wealth)  ∘ TimeDiscount
```

The **one** change versus `examples/default` is the `Savings` step. A
`ConsumptionSavingsStage` can only express a **unit**-price budget
`c = before − after`; it sees the chosen `a'` only through `c`, so it cannot
apply a destination price `q(a', y)·a'`. Instead `Savings` is a **raw**
`ArgmaxStage(:wealth)` whose reward is a plain **closure** `(; income, status,
env) -> M` building the `(after, before)` matrix

```
M[after, before] = u_crra( x − q(a', income)·a' ),     a' = grid[after],  x = grid[before]
```

with `a' ≥ 0` priced risk-free, infeasible `c ≤ 0` gated out, and the excluded
branch (`status = 2`) forced to autarky `a' = 0`. The discount is a separate
`∘ TimeDiscountingStage(β)` (the argmax carries no `β`). No reward struct, no
`src/` edit — the reward is a closure, exactly as the framework permits.

Everything **outside** the block is custom driver code in `steady_state.jl`:
the bond-price schedule and its fixed-point loop.

## The outer loop (custom driver)

A damped fixed point on the whole `q` schedule (an `(N_a, N_y)` matrix), the
schedule analogue of the `examples/aiyagari` tatonnement on a scalar `K`:

1. guess `q` → solve the household `V`/`Λ` at `env = (; ŷ, rf, q)`;
2. read the per-cell repay/default policy off the solved `DefaultStage`
   (`policy(::ArgmaxStage)` — ordinary "reach into the solved block" driver work);
3. reprice `q'(a',y) = (1/(1+rf))·(1 − Σ_{y'} P[y,y']·default(a',y'))`;
4. damp `q ← q + speed·(q' − q)` and repeat to a fixed point.

## Run

```
julia --project=. examples/arellano_default/steady_state.jl
```

At the default calibration (`β = 0.915`, `σ = 2`, `rf = 0.017`, `ŷ = 0.95·E[y]`,
`ψ = 0.28`; a 5-state Rouwenhorst endowment, `N_a = 120` asset grid concentrated
in the debt region) the `q` fixed point converges in ~30 iterations and shows the
canonical Arellano picture:

- **Spreads rise with debt**: ~0 bps at `a' ≥ 0`, ~1400 bps near the borrowing
  path (`a' ≈ −0.10`), exploding as debt deepens.
- **Countercyclical default**: the default frontier (least debt that still
  triggers default) is far smaller in low-income states than high — the poorest
  state defaults at trivial debt, the richest never.
- An on-path default rate of ~6% per period with ~13% of mass in exclusion.

The asymmetric cost `min(y, ŷ)` is essential: a *proportional* cost `(1−λ)·y`
scales the debt-relief benefit and the default cost together, flattening the
frontier across income and killing on-path default. Capping output at `ŷ` makes
default cheap when poor and costly when rich — the device behind countercyclical
default and graded spreads.
