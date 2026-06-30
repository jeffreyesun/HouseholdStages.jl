# Endogenous separations (Mortensen–Pissarides quit leg)

A dogfooding example: a search-and-matching household with an ENDOGENOUS-quit margin,
expressed as a pure composition of EXISTING HouseholdStages library stages, with no
bespoke household stage rolled here.

## Mechanism

State `(:z, :emp)`. `z` is match productivity (a persistent Rouwenhorst-discretized
AR(1) chain); `:emp` is the 2-level labor state. An employed worker whose match
productivity `z` has deteriorated chooses whether to **keep** the match or **quit**
into unemployment. This endogenous-quit margin sits on top of the standard
search-matching inflow (the unemployed choose search effort and find jobs) and an
exogenous separation rate `δ`. Quits occur at low `z` — the Mortensen–Pissarides
(1994) reservation-productivity margin. The **total** separation rate is the
exogenous `δ` plus the endogenous quit flow.

The quit margin is produced by an `ArgmaxStage` gating trick. The reward matrix
`M[after, before]` on the 2-level `:emp` axis uses `-Inf` to forbid self-promotion
(unemployed → employed — hiring is the matching stage's job), so the only genuine
choice is the employed worker's keep-vs-quit. The `:z` axis is a **spectator**: it
makes the employed continuation `V_end[emp]` rise with `z` while the unemployed
continuation `V_end[unemp]` is flat, so the quit index flips below a reservation `z`.

## The exact `∘` chain

`∘` runs the LEFT stage FIRST in time order:

```
QuitChoice ∘ FlowUtility ∘ Discount ∘ ZShock ∘ Matching
```

| stage         | library stage                              | role |
|---------------|--------------------------------------------|------|
| `QuitChoice`  | `ArgmaxStage(axis = :emp)`                 | gated (max,+) keep/quit; `M[emp,unemp] = -Inf` forbids self-promotion |
| `FlowUtility` | `UtilityStage`                             | `u(w·z)` employed (wage rises in `z`), `u(b_u)` unemployed |
| `Discount`    | `TimeDiscountingStage(β)`                  | `V_start = β·V_end` |
| `ZShock`      | `MarkovStage(axis = :z)`                   | persistent match-productivity chain (Rouwenhorst AR(1)) |
| `Matching`    | `SearchMatchingStage(axis = :emp)`         | unemployed search (effort, cost, job-finding at fixed `θ`); employed separate exogenously at `δ` |

The transition matrix `T_z`, the `z` level grid, and the gated reward matrix are
plain economic DATA built by helper functions (`rouwenhorst`, `z_process`) — exactly
the intended division of labor (data feeds existing stages; no new stage). The search
effort grid, effort-cost, and job-finding closures are fed to the existing
`SearchMatchingStage`.

This example PAIRS with `examples/mccall_search`: both use the same gated
`ArgmaxStage(:emp)` `-Inf` trick with a spectator axis driving a threshold, in
OPPOSITE directions — there the unemployed choose to ACCEPT (→ employed); here the
employed choose to QUIT (→ unemployed).

## Fidelity note

This captures the Mortensen–Pissarides endogenous-separation (reservation-
productivity) leg faithfully: persistent idiosyncratic match productivity, an
employed worker's optimal quit threshold, layered on a search-matching inflow and an
exogenous separation rate. Two deliberate stylizations: (i) market tightness `θ` is
FIXED (partial equilibrium — no free-entry / vacancy-posting block, unlike
`examples/search_matching`); (ii) new hires inherit their current `z` cell rather than
drawing from a hiring distribution, since `SearchMatchingStage` operates only on the
`:emp` axis (`z` rides along as a spectator). Both are clean stylizations, not
workarounds — the quit mechanism itself is exact.

## Status

Solves cleanly. With the default calibration (`β = 0.96`, `σ = 2`, `w = 1`,
`b_u = 0.75`, exogenous `δ = 0.05`, `θ = 2.5`, a 5-state Rouwenhorst `z` chain with
`ρ = 0.90`, `σ = 0.20`, normalized to `E[z] = 1`):

- `ΣΛ = 1.000000` (mass conserved)
- employment ≈ 0.648
- exogenous separation `δ = 0.050`; endogenous quit rate ≈ 0.045
- **total separation rate ≈ 0.093** (= δ + (1−quit)·... composition of both margins)
- quit threshold: employed quit at the two lowest `z` (`z ≲ 0.57`); reservation
  productivity is interior
- mean `z` among the employed ≈ 1.20 > unconditional `E[z] = 1.00` (positive
  selection — quits prune low-productivity matches)
- quit policy verified to be a low-`z` prefix (monotone)

Run: `julia --project=. examples/endogenous_separations/steady_state.jl`.
