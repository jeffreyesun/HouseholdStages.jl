# Menu-cost price setting — Golosov–Lucas (2007) / Nakamura–Steinsson (2010)

An (S,s) **pricing band**, built as a `∘` composition of existing library stages, read on the
firm's **relative price** axis. A firm posts a nominal price; aggregate inflation erodes its
*relative* price each period, and re-pricing costs a fixed **menu cost** `F`. The firm therefore
sits in an inaction band, letting its price drift down, and **resets** only when the gap to its
frictionless optimum is large enough to justify paying `F`.

## The dictionary mapping

This is the same keep-vs-adjust object as the lumpy-investment (S,s) capital band
(`examples/lumpy_investment`), read on a different axis:

| household / capital reading | menu-cost reading |
|---|---|
| wealth `b` / capital `k` (the operative axis) | the firm's **relative (real) price** `p` — a stock that erodes with inflation |
| income / productivity shock | idiosyncratic **marginal-cost** shock `z` (Rouwenhorst AR(1)) |
| (S,s) durable purchase / lumpy investment | menu-cost **price reset** (pay `F`, jump to a new price) |
| depreciation / a deterministic `WealthChange` drift | **inflation erosion** of the relative price — a deterministic Markov down-shift |

The relative price `p` behaves exactly like a stock that depreciates: doing nothing lets it slide,
and "topping it back up" pays a fixed cost. The marginal-cost shock plays the role of the income
shock, and the reset decision is the (S,s) durable-adjustment ArgmaxStage.

## The exact chain

```
Shock ∘ Erosion ∘ Profit ∘ Reset ∘ Discount
= MarkovStage(:z)
∘ MarkovStage(:price; deterministic down-shift)        # inflation erosion
∘ UtilityStage(profit(p,z) = D·(p^{1−ε} − z·p^{−ε}))   # monopolistic flow profit
∘ ArgmaxStage(:price; reward M[p',p] = −F·1{p'≠p})     # (S,s) keep/reset
∘ TimeDiscountingStage(β = 1/(1+r))
```

Five existing exported stages, normally parameterized — **no bespoke stage, no custom kernel**.

- **Erosion** is a `MarkovStage` on the `:price` axis whose transition is a deterministic down-shift
  permutation `T[i, max(i−1,1)] = 1`. The log-price grid spacing is set to the inflation step
  `log(1+π)`, so one period of erosion is exactly a **one-index shift down** (index 1 absorbing).
  This is a normal MarkovStage parameterization — a 0/1 shift matrix — playing the role depreciation
  plays for a capital stock.
- **Profit** is single-peaked in `p` with frictionless optimum (the static markup)
  `p*(z) = (ε/(ε−1))·z`. It reads both axes, so it lives in its own `UtilityStage`; the reset reward
  sees only the `(p', p)` price pair.
- **Reset** is the (S,s) ArgmaxStage: keeping the price (`p' = p`, the diagonal) is free, resetting to
  any other grid price pays `F`. The argmax is brute — the fixed cost makes the reward
  non-supermodular, so a monotone solve would not apply.

The backward sweep reproduces the menu-cost Bellman

```
V(p,z) = E_{z'|z}[ profit(p_e, z') + max_{p'} ( −F·1{p'≠p_e} + β·V(p', z') ) ],   p_e = erode(p).
```

## The monetary block is the outer loop (not modeled here)

The aggregate **price level**, the **inflation rate** `π`, and the cross-sectional price distribution
as an aggregate state are treated as partial-equilibrium-**exogenous** — a single stationary solve at a
fixed `π`, exactly as the rental rate is exogenous for lumpy investment and the interest rate for
Aiyagari. Closing the model (a Calvo-vs-menu-cost Phillips curve, monetary policy) would wrap this
firm block in a fixed point on `π` and the price level. That is the caller's loop, deliberately
outside the agent block.

## Running it

```
julia --project=. examples/menu_cost/steady_state.jl
```

`steady_state.jl` reports V finiteness, mass conservation, mean price / markup / profit, the spread of
the price-marginal (a non-degenerate band), and the headline **frequency of price change** — the
Nakamura–Steinsson moment, computed by re-deriving the keep/reset policy from the solved `V` over the
post-shock, post-erosion population facing the choice.

At the default parameters (`ε = 4`, `F = 0.025`, `π = 0.03`, `r = 0.04`):

- frequency of price change ≈ **0.16** per period (the rest sit in the inaction band);
- the price-marginal carries mass on **14 of 30** grid points (a genuine band, not a degenerate point);
- mean realized markup ≈ **1.39**, slightly above the frictionless `4/3 ≈ 1.333` — firms reset *above*
  their static optimum because they anticipate the downward inflation drift (the standard menu-cost
  markup bias);
- reset targets rise monotonically in the marginal cost `z`.

## Literature

Sheshinski–Weiss (1977), the (S,s) pricing band; Golosov–Lucas (2007 JPE); Nakamura–Steinsson
(2010 QJE), the frequency-of-price-change moment; Barro (1972), the band-width / menu-cost balance.
The convex-cost (no-band) counterpart is the Calvo / quadratic-adjustment model; the capital-axis
analogue of this band is `examples/lumpy_investment`.
