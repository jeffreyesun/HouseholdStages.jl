# Becker–Tomes (1979, 1986) — child human-capital investment

Altruistic parents invest in their child's **human capital** and leave a
**financial bequest**, generating intergenerational persistence. A dynasty is
recursive with the *same* value function each generation, so the "period" is a
**generation** and the model is a **stationary infinite-horizon** problem over two
endogenous states — wealth `b` and human capital `h`.

## Household block (existing stages only)

Two coupled continuous choices (`h'`, `b'`) financed from one budget → the
auxiliary-choice-axis pattern (as in `two_asset_hank`). Time order:

    Earn ∘ [ ChooseChildHC ∘ DebitInvestment ∘ CommitChildHC ∘ Forget ]
         ∘ Reproduce ∘ ConsumeBequeath

- `Earn`            — `IncomeStage(:wealth, income_axis=:h)`: `b ↦ (1+r)b + w·h`.
- `ChooseChildHC`   — `ArgmaxStage` picks child HC `h'` onto the auxiliary `:hc`
  axis (reward 0; benefit via continuation, cost debited downstream).
- `DebitInvestment` — `WealthChangeStage(:wealth)` debits child-HC cost
  `κ·h'^θ / h^ψ` (reads `:hc`, `:h`, `:wealth`); cost falls in parent `h`.
- `CommitChildHC`   — `WealthChangeStage(:h)` writes `:h ← h'` (child's HC is the
  next generation's state).
- `Forget`          — `ForgetfulSumStage(:hc)` collapses the auxiliary axis.
- `Reproduce`       — `ReproductionStage(s = fertility)` scales mass for fertility.
- `ConsumeBequeath` — `ConsumptionSavingsStage(:wealth)` with `β = α` (altruism);
  bequest `b' = wealth − c`, continuation = child's dynastic value.

No bespoke stage. The cross-generation closure **is** the stationary fixed point:
the chosen `(b', h')` seeds the next generation as `(b, h)`, and
`solve_steady_state_given_env!` finds the dynastic value `V(b,h)` and the stationary
distribution of dynasties `Λ(b,h)`. No custom cohort iteration is needed.

## Two persistence channels

1. **Financial bequest** `b'` (active only when `α(1+r)` is high enough that parents
   bequeath rather than consume everything — otherwise the wealth axis goes inert).
2. **Child-HC cost** `κ·h'^θ / h^ψ`, falling in parental `h` (`ψ>0`): high-`h`
   parents produce child HC more cheaply.

Convexity `θ>1` gives diminishing returns to child HC, so `h` mean-reverts and a
stationary distribution exists — the ◐ content is exactly this cross-generation
mapping + the recursive driver.

## Run

    julia --startup-file=no --project=. examples/becker_tomes/steady_state.jl

### Output (confirms it solves + persistence)

```
Becker–Tomes dynastic steady state (σ = 2.0, α = 0.70, θ = 2.10, ψ = 0.45)
  mass(Λ)                        = 1.000000  (fertility = 1.00)
  V finite everywhere            = true
  VFI iters / Λ iters            = 45 / 400
  mean wealth (bequest)          = 4.8353
  mean human capital             = 3.3929
  E[h' | h] at h = 2.48 / 4.55 / 6.63 = 3.069 / 3.664 / 4.364
  intergenerational persistence  = true
  intergenerational HC elasticity= 0.358
```

`V` finite, `Λ` stationary (mass conserved at fertility 1), both endogenous states
active (mean bequest 4.84 and mean HC 3.39 both positive), and intergenerational
persistence confirmed: children of high-`h` parents have higher `h` (`E[h'|h]`
rising), with an intergenerational HC elasticity of 0.36 — squarely in the
empirical range.

## Reuse for the rest of the child-HC family

This block is the canonical child-HC dynasty. The neighbouring literature is the
**same** household block with longer ∘-chains / more driver structure:

- **Caucutt–Lochner** (two-phase early/late childhood HC): two
  `choose/debit/commit` triples (one per childhood phase) instead of one, before the
  consume/bequeath margin — a longer chain of the same stages.
- **Lee–Seshadri** (many-phase): one such triple per phase.
- **Daruich** (public transfers): the same block plus a transfer term in the
  `Earn`/budget step (a `WealthChangeStage` or an `env` transfer), no new stage.

`becker_tomes` (the dynasty + aux-axis coupling) together with `manuelli_seshadri`
(the multi-phase HC schedule) demonstrate both ingredients those models need, so the
pattern is reusable for them.
