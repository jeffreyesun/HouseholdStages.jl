# Hopenhayn (1992) — firm entry & exit

A firm is an **agent block** under the §6 household↔firm dictionary; nothing in the
`V`/`Λ` machinery knows household from firm. This example reads the demographic stages
as firm turnover:

| Household reading | Firm reading | Stage |
|---|---|---|
| income / employment shock | productivity shock `z` | `MarkovStage(:z)` |
| death | firm **exit** | `EndogenousExit` |
| bequest / value of death | **scrap / liquidation value** | the required `bequest` field |
| birth | firm **entry** | `EntryStage` (the additive `Λ += g` source) |
| consumption / flow felicity | operating profit `π(z)` | `UtilityStage` |

## The block (five existing stages, no bespoke stage)

```
Entry ∘ Profit ∘ Exit ∘ Discount ∘ Shock
= EntryStage(g) ∘ UtilityStage(π(z)) ∘ EndogenousExit(scrap) ∘ TimeDiscountingStage(β) ∘ MarkovStage(:z)
```

The backward (value) sweep runs the chain right-to-left and reproduces the Hopenhayn
recursion exactly:

```
V(z) = π(z) + max{ scrap , β·E[V(z')|z] }.
```

`Shock` forms the continuation expectation `E[V(z')|z]`; `Discount` scales by `β =
1/(1+r)`; `Exit` is the optimal-stopping `max(continuation, scrap)` — the §5(i)
keep-vs-stop `ArgmaxStage` that the exit composite wraps over a transient `:exiting`
axis (declared at size 1, grown to 2, collapsed back); `Profit` adds the per-period
operating profit (the static labour choice is closed-form-substituted, so productivity
`z` is the only firm state); `Entry` adds the entrant inflow `M·ν`.

Forward, entrants are seeded, survive-or-exit by the seated stopping rule (firms whose
discounted continuation falls below scrap leave — including entrants who draw too low a
`z`), and survivors' productivity transitions. **Mass is not conserved** (entry in, exit
out); the stationary firm mass settles at entrant-inflow / exit-rate.

## What is the outer loop (the caller's, never the block)

Exactly as for Aiyagari: the **free-entry condition** `∑_z ν(z)·V(z) = c_e` pins the
equilibrium wage/price, and aggregate clearing pins the entrant mass `M`. Both are scalar
fixed points; `free_entry_residual` (model.jl) is the object to root on the wage. Here we
solve the firm block at given prices (partial equilibrium) and report the stationary firm
distribution, exit rate, and selection.

## Running it

```julia
julia --project=. examples/hopenhayn/steady_state.jl
```

At the default calibration the block solves to a finite value everywhere, a stationary
firm mass ≈ 11.3, an exit rate ≈ 5.6%, and survivor selection lifts mean productivity
(≈ 1.46) above the entrant mean (≈ 1.0) — low-`z` firms exit, the survivors are
positively selected.

## Variant (covered by pattern, not separately built)

**Hopenhayn–Rogerson (1993)** adds a firing cost for employment-protection policy: this
is the Hopenhayn chain with an extra `ArgmaxStage(:n over {keep, adjust})` carrying the
firing cost on an employment axis (the §5(i) keep/adjust shape of
`examples/lumpy_labor`). A one-stage extension on the same chain.
