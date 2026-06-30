# McCall (1970) reservation-wage search

A dogfooding example: the canonical sequential-search model expressed as a pure
composition of EXISTING HouseholdStages library stages, with no bespoke household
stage rolled here.

## Mechanism

State `(:wage, :emp)`. An unemployed worker holds a wage OFFER drawn from a
distribution `F` over the wage grid and chooses to **accept** it (become employed at
that wage forever, until separated) or **reject** it (stay unemployed at benefit
`b_u` and redraw a fresh offer next period). The employed separate exogenously at
rate `δ`. The optimal policy is a **reservation wage** `w*`: accept iff `w ≥ w*`.

The accept margin is produced by an `ArgmaxStage` gating trick. The reward matrix
`M[after, before]` on the 2-level `:emp` axis uses `-Inf` to forbid the voluntary
quit (employed → unemployed), so the only genuine choice is the unemployed worker's
accept-vs-reject. The `:wage` axis is a **spectator**: it makes the employed
continuation `V_end[emp]` rise with the held wage while the unemployed continuation
`V_end[unemp]` is flat (the unemployed redraw), so the accept index flips at `w*`.

## The exact `∘` chain

`∘` runs the LEFT stage FIRST in time order:

```
AcceptReject ∘ FlowUtility ∘ Discount ∘ Separation ∘ OfferDraw
```

| stage          | library stage                       | role |
|----------------|-------------------------------------|------|
| `AcceptReject` | `ArgmaxStage(axis = :emp)`          | gated (max,+) accept/reject; `M[unemp,emp] = -Inf` forbids quitting |
| `FlowUtility`  | `UtilityStage`                      | `u(w)` employed, `u(b_u)` unemployed |
| `Discount`     | `TimeDiscountingStage(β)`           | `V_start = β·V_end` |
| `Separation`   | `MarkovStage(axis = :emp)`          | employed → unemp w.p. `δ` (exogenous job destruction) |
| `OfferDraw`    | `MarkovStage(axis = :wage)`         | unemployed redraw from `F` (every row = `F`); employed keep wage (identity) |

The two `MarkovStage` kernels, the gated reward matrix, and the offer distribution
`F` are plain economic DATA built by helper functions (`offer_distribution`,
`offer_kernel`, `identity_kernel`, `wage_grid`) — exactly the intended division of
labor (data feeds existing stages; no new stage).

This example PAIRS with `examples/endogenous_separations`: both use the same
`ArgmaxStage(:emp)` `-Inf`-gating trick with a spectator axis driving a threshold, in
OPPOSITE directions — here the unemployed choose to ACCEPT (→ employed); there the
employed choose to QUIT (→ unemployed).

## Fidelity note

This is faithful textbook McCall (Ljungqvist & Sargent ch. 6): sequential search
over i.i.d. offers, accept/reject with an exogenous separation rate, CRRA flow
utility. The only stylization is that the offer grid is finite (a fine 60-point
grid), so `w*` is read as the lowest accepted grid point rather than solved as a
continuous indifference root. Partial equilibrium: `b_u` is exogenous, no market to
clear, one inner V/Λ solve.

## Status

Solves cleanly. With the default calibration (`β = 0.96`, `σ = 2`, `b_u = 0.70`,
`δ = 0.10`, offers log-normal with median 1.0):

- `ΣΛ = 1.000000` (mass conserved)
- employment ≈ 0.776
- reservation wage `w* ≈ 1.14` (interior; above `b_u` by the option value of search)
- mean accepted wage ≈ 1.48 (> `w*`, as only high offers are accepted)
- accept policy verified monotone (non-decreasing) in the wage

Run: `julia --project=. examples/mccall_search/steady_state.jl`.
