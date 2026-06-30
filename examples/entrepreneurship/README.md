# Entrepreneurship & wealth inequality (Quadrini 2000; Cagetti–De Nardi 2006)

Occupational choice — worker vs (productive, risky, limited-liability) **entrepreneur** — as the
engine of wealth concentration. High-productivity households sort into entrepreneurship, capture the
productivity boost on their invested wealth, and accumulate into the top tail. **Catalog status: ◐.**
This folder is BOTH a clean build *and* a precise give-up: the literal occupational form is not a
straight composition, but the uniform-leg reformulation the catalog names *is*, and it solves.

## The give-up — why the literal form is not a straight composition

The natural occupational model has a worker spine and an entrepreneur spine with an **extra** risk
stage:

```
worker_leg       = Receipt ∘ Savings                          # 4 expanded stages
entrepreneur_leg = Receipt ∘ Savings ∘ Business(MeanVariance)  # 5 expanded stages
product(worker_leg, entrepreneur_leg; axis = :occupation)      # ← REJECTED
```

`product`/`⊕` requires every leg to have the **identical concrete Spec type** —
`ProductStageSpec` asserts `all(s -> typeof(s) === first_type, components)`. A `ChainStage` is
`ChainStageSpec{Stages<:Tuple}`, **parameterised on the tuple of its component spec types**, so a
4-stage chain and a 5-stage chain are *different concrete types*. The assertion fails (verified):

```
AssertionError: all((s -> typeof(s) === first_type), components)
```

This is the documented ◐: **distinct stage CHAINS per leg** — the entrepreneur having a stage the
worker lacks — is per-axis gating the library deliberately does not offer. (The single-chain
`:occupation`-axis route used by `examples/indivisible_labor`, where downstream `WealthChangeStage`
closures *read* the chosen indicator, also cannot help here: the entrepreneur's distinguishing
feature is the `MeanVarianceStage` itself, and a `MeanVarianceStage`'s risky technology
`shares`/`risky_returns`/`probs` are NOT axis-closures — a single shared business stage offers the
gamble to *all* occupations and cannot be switched on for one slice and off for another.)

## The clean exit that DOES solve — a fully separate `⊕` leg

The catalog names two clean exits, "a wealth-floor or a fully separate `⊕` leg." This example takes
the **separate-`⊕`-leg** exit. Make the two legs **structurally uniform** by giving BOTH the same
chain and letting the worker carry a *degenerate* `Business` (a `MeanVarianceStage` whose risky
returns equal the risk-free return — a safe asset) and the identity branch of `Realize`:

```
OccChoice ∘ ⊕_occupation{ worker_leg, entrepreneur_leg }

each leg:  Receipt ∘ Savings ∘ Business ∘ Realize ∘ ZShock
```

| stage | library stage | does |
|---|---|---|
| `OccChoice` | `ArgmaxStage` (`:occupation`) | per `(wealth, z)`, pick the occupation with higher continuation, less `κ` |
| `Receipt` | `WealthChangeStage` | `a ↦ a + wage` (worker wage `w`; entrepreneur `w_e < w`) |
| `Savings` | `ConsumptionSavingsStage` | pick the stake `a'`, `c = x − a'`, CRRA |
| `Business` | `MeanVarianceStage` | worker: degenerate safe asset; entrepreneur: mean-neutral two-point project, intensity θ |
| `Realize` | `WealthChangeStage` | entrepreneur `a ↦ max(z·a, a_floor)`; worker `a ↦ max(a, a_floor)` |
| `ZShock` | `MarkovStage` (`:z`) | productivity transitions for next period |

The legs differ **only in captured VALUES** (`wage`, `business_returns`) and a captured **Bool**
(`entrepreneur`, branching the shared `Realize` closure). Because every closure is defined at one
syntactic site (a single `entrepreneurship_leg` builder), the closure *types* match and the two
`ChainStageSpec` types are identical — exactly the `examples/discount_heterogeneity` device (one
builder, parameterised by value). `product` accepts them; the `ArgmaxStage` occupational choice
composes on top and the whole block solves to a stationary distribution. **No bespoke stage
anywhere.** The Vereshchagina–Hopenhayn **wealth-floor** in `examples/risk_shifting` is the OTHER
clean exit (it captures the entrepreneurial risk margin gating-free, with no occupation axis).

## What the constant-returns `MeanVarianceStage` forces (honest)

A streaming `MeanVarianceStage` is **constant-returns**, and its returns cannot read the
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

- **A small entrepreneurial elite.** Entrepreneur population share `≈ 0.034` — a few percent, as in
  the data and in Cagetti–De Nardi.
- **Wealth concentration.** Top-10% wealth share `≈ 0.63`; entrepreneurs hold `≈ 2.5×` the worker's
  mean wealth. The productive entrepreneurs are the top tail.
- **Comparative static.** A steeper business success multiple `R_up` pulls slightly more households
  into entrepreneurship (`share 0.037 → 0.045` for `R_up = 1.45 → 1.70`).

The occupation-axis moment and the direct `Λ`-slice agree (`0.034` both ways), validating the
`:occupation` accounting. Partial equilibrium — no market clears, so the outer loop is a single
`solve_steady_state_given_env!`.

## Literature

Quadrini (2000, RED); Cagetti & De Nardi (2006, JPE); Vereshchagina & Hopenhayn (2009, AER) for the
risk-margin convexity (`examples/risk_shifting`). MODEL_CATALOG §2 (entrepreneurship ◐) and §4
(occupational choice as direction-control sugar).
