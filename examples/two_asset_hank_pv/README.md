# Two-asset HANK (Kaplan–Moll–Violante 2018) — Route A′ (portfolio value)

Same economics as `examples/two_asset_hank`, a **cleaner state**. A household holds a liquid asset `b`
(low return `r_b`, free to adjust) and an illiquid asset `a` (high return `r_a`, costly to adjust).
Each period it picks an illiquid target `a'` (paying a convex cost on the net deposit
`d = a' − (1+r_a)a`) and consumes/saves the liquid balance.

## The reformulation

Instead of two stock axes `(b, a)`, track

```
W = b + a   (portfolio value)   and   a   (illiquid),   with liquid  b = W − a  DERIVED.
```

A pure transfer between the stocks (the deposit) **conserves `W`**, so the illiquid choice `a'` moves
**only** the `:illiquid` axis — an ordinary single-axis argmax. `W` is untouched by the rebalance, and
liquid `= W − a'` automatically reflects the deposit. There is **no auxiliary `:illiquid_choice` axis
and no `n_choice×` memory** (the whole point — contrast the auxiliary-axis version below).

## Household block (existing stages only, no bespoke stage, no auxiliary axis)

```
IncomeShock ∘ ReturnsW ∘ ReturnsA ∘ Rebalance ∘ Consume
```

| stage | library stage | role |
|---|---|---|
| `IncomeShock` | `MarkovStage(:income)` | income draw |
| `ReturnsW` | `WealthChangeStage(:wealth)` | `W ↦ (1+r_b)W + (r_a−r_b)a + w·y` (reads OLD `a`, so time-before `ReturnsA`) |
| `ReturnsA` | `WealthChangeStage(:illiquid)` | `a ↦ (1+r_a)a` |
| `Rebalance` | `ArgmaxStage(:illiquid)` | choose `a'`, reward `M[a',a] = −χ(a'−a)`; moves **only** `:illiquid`, `W` untouched |
| `Consume` | `ContinuousArgmaxStage(:wealth)` | choose `W'`, `c = W − W'`, `u(c)`; liquid constraint `W' ≥ a` masked into the reward; β here |

`ReturnsW` uses `b = W − a` so the gross receipts `(1+r_b)(W−a) + (1+r_a)a + w·y` collapse to
`(1+r_b)W + (r_a−r_b)a + w·y`. The `Consume` reward masks `W' < a` (so liquid `= W' − a ≥ 0`) to `−Inf`,
with a subsistence floor `ε` on consumption so the constrained grid bottom keeps a feasible action.

## The utility-cost reading (the one residual)

The deposit cost `χ(d)`, `d = a' − (1+r_a)a`, depends on **both** the chosen `a'` and the old
(post-return) `a`. In Route A′ the rebalance **overwrites** the illiquid axis with `a'`, so a downstream
*resource* debit would no longer see the old `a`. We therefore charge `χ` as a **utility cost** inside
the rebalance reward `M[a', a]` — the reward matrix sees both endpoints at once, so no overwrite arises.
Making `χ` a resource cost (debit `W` by `χ`) would reintroduce that small overwrite; the
auxiliary-axis version (`examples/two_asset_hank`) takes the resource-cost reading via its extra axis.
This is the documented, accepted difference between the two examples.

## Contrast with the auxiliary-axis version (`examples/two_asset_hank`)

| | `two_asset_hank` (auxiliary axis) | `two_asset_hank_pv` (Route A′, this) |
|---|---|---|
| state | `(b, a, income, illiquid_choice)` | `(W, a, income)` — no auxiliary axis |
| illiquid choice | `ArgmaxStage` onto a separate `:illiquid_choice` axis, then a `WealthChangeStage` debits liquid by `d`, a second credits illiquid, a `ForgetfulSumStage` collapses the axis | a single `ArgmaxStage` on `:illiquid` (`W` is conserved, liquid is derived) |
| intermediate tensor | `N_a×` larger (the choice grid) | no blow-up |
| deposit cost `χ` | **resource** cost (debit liquid) | **utility** cost (in the argmax reward) |

Both implement the same KMV economics from existing stages with no bespoke household stage.

## Verified

`test/test_example_two_asset_hank_pv.jl`:

1. the novel `Rebalance ∘ Consume` block's backward value equals its brute `(W, a)` Bellman **to
   machine precision** (max abs diff `0.0` — both argmaxes choose grid indices and read continuations
   on-grid, so the block carries no interpolation to match);
2. the full model solves to a stationary steady state (mass `≈ 1`, finite `V`, illiquid share in
   `(0, 1)`).

Default-parameter steady state: mean liquid `≈ 0.88`, mean illiquid `≈ 10.1` (grid-interior),
illiquid share `≈ 0.92` — the KMV stylized fact that liquid is a small transaction buffer while most
wealth sits illiquid. `β(1+r_a) < 1` keeps the illiquid stock off the grid ceiling.

## Caveats

- The `Consume` choice uses the monotone (`:divide_conquer`) walk with `assume_monotone = true`; the
  brute test (max abs diff `0.0` across seeds) confirms the optimal policy is monotone on these grids.
- Quadratic-cost (smooth) adjustment; a fixed cost would give the lumpy (S,s) margin.

## Run

```julia
julia --project=. examples/two_asset_hank_pv/steady_state.jl
```
