# Entrepreneurship & wealth inequality (Quadrini 2000; Cagetti–De Nardi 2006)

Occupational choice — worker vs (productive, risky, limited-liability) **entrepreneur** — as the
engine of wealth concentration. High-productivity households sort into entrepreneurship, capture the
productivity boost on their invested wealth, and accumulate into the top tail. **Catalog status: ◐.**
The occupational margin builds as a straight composition; the ◐ is one thing the library cannot say,
named below.

## Why both legs carry a `Business` stage

`⊕`/`product` asks its legs to share a start layout and an end layout and leaves their specs free, so
a worker spine and an entrepreneur spine with an extra risk stage join as written. The legs here are
structurally uniform anyway, for a modelling reason: the worker's asset earns the safe return `R_f`,
and that return **is** the `Business` stage's `anchor`. Dropping the stage from the worker leg would
drop the safe return with it, so the worker carries a *degenerate* `Business` — zero excess mean, a
numerically-negligible excess sd (the Gaussian stage requires `σ > 0`) — and the identity branch of
`Realize`.

What the occupational margin cannot be built from is ONE shared business stage gated on the
occupation axis. A `GaussianLoadingStage`'s risky technology `anchor`/`increment_mean`/`increment_sd`
are scalars or `FromEnv`, never axis-closures, so a single business stage offers the same gamble to
every occupation and cannot be switched on for one slice and off for another. The risk margin has to
live in a per-leg stage — which is what the `⊕` build gives it, and which is the catalog's ◐
(MODEL_CATALOG §2, limitation (a)).

## The block — one `⊕` leg per occupation

The catalog names two routes to the entrepreneurial risk margin, "a wealth-floor or a fully separate
`⊕` leg." This example takes the **separate-`⊕`-leg** route:

```
OccChoice ∘ ⊕_occupation{ worker_leg, entrepreneur_leg }

each leg:  Receipt ∘ Savings ∘ Business ∘ Realize ∘ ZShock
```

| stage | library stage | does |
|---|---|---|
| `OccChoice` | `ArgmaxStage` (`:occupation`) | per `(wealth, z)`, pick the occupation with higher continuation, less `κ` |
| `Receipt` | `WealthChangeStage` | `a ↦ a + wage` (worker wage `w`; entrepreneur `w_e < w`) |
| `Savings` | `ConsumptionSavingsStage` | pick the stake `a'`, `c = x − a'`, CRRA |
| `Business` | `GaussianLoadingStage` (the portfolio stage read as a business-risk dial: anchor = `R_f`, increment = the project's excess payoff) | worker: degenerate safe asset; entrepreneur: mean-neutral Gaussian project (moment-matched to the succeed/fail bet), continuous intensity θ |
| `Realize` | `WealthChangeStage` | entrepreneur `a ↦ max(z·a, a_floor)`; worker `a ↦ max(a, a_floor)` |
| `ZShock` | `MarkovStage` (`:z`) | productivity transitions for next period |

The legs differ **only in captured VALUES** (`wage`, the excess moments `(excess_mean, excess_sd)`)
and a captured **Bool** (`entrepreneur`, branching the shared `Realize` closure) — one
`entrepreneurship_leg` builder parameterised by value, exactly the `examples/discount_heterogeneity`
device. They share their two boundary layouts, which is what `⊕` asks of its legs; the `ArgmaxStage`
occupational choice composes on top and the whole block solves to a stationary distribution. **No
bespoke stage anywhere.** The Vereshchagina–Hopenhayn **wealth-floor** in `examples/risk_shifting` is
the other route to the same margin (it captures the entrepreneurial risk margin with no occupation
axis at all).

## What the constant-returns `GaussianLoadingStage` forces (honest)

A streaming `GaussianLoadingStage` is **constant-returns**, and its returns cannot read the
productivity / occupation axis. Three dead ends follow, and they shaped the calibration:

1. A mean **premium** (`E[R_business] > R_f`) lets the patient rich accumulate without bound — no
   stationary distribution (the Λ solve diverges).
2. **Mean-neutral**, the project's option value lives only at the limited-liability floor; away from
   the floor the occupational value gap collapses to the *constant* wage gap, so the choice is
   **all-or-nothing** (a knife-edge: everyone or no one enters).
3. The literature's bounding force — **decreasing returns to entrepreneurial scale** (span of
   control, the productive-but-self-limiting business) — is **not expressible** in a streaming
   constant-returns stage.

The faithful, library-only resolution puts the entrepreneur's edge in a **transient productivity
state** `z`: `Realize` compounds an entrepreneur's invested wealth by `z` (a productive entrepreneur
grows fast), but `z` mean-reverts to its low state, so the long-run multiplier is `< 1` and the
distribution is **stationary** despite transient high-`z` accumulation. High-`z` households strictly
prefer entrepreneurship (to capture the boost) — a genuine *state-dependent* occupational margin, not
a knife-edge — and they form the wealth tail. This is the Cagetti–De Nardi mechanism (entrepreneurial
ability + self-financed accumulation), realised within the streaming vocabulary.

## What the run shows

`julia --project=. examples/entrepreneurship/steady_state.jl`:

- **A small entrepreneurial elite.** Entrepreneur population share `≈ 0.033` — a few percent, as in
  the data and in Cagetti–De Nardi.
- **Wealth concentration.** Top-10% wealth share `≈ 0.64`; entrepreneurs hold `≈ 2.5×` the worker's
  mean wealth. The productive entrepreneurs are the top tail.
- **Comparative static.** A steeper business success multiple `R_up` pulls slightly more households
  into entrepreneurship (`share 0.035 → 0.043` for `R_up = 1.45 → 1.70`).

The occupation-axis moment and the direct `Λ`-slice agree (`0.033` both ways), validating the
`:occupation` accounting. Partial equilibrium — no market clears, so the outer loop is a single
`solve_steady_state_given_env!`.

## Literature

Quadrini (2000, RED); Cagetti & De Nardi (2006, JPE); Vereshchagina & Hopenhayn (2009, AER) for the
risk-margin convexity (`examples/risk_shifting`). MODEL_CATALOG §2 (entrepreneurship ◐) and §4
(occupational choice as direction-control sugar).
