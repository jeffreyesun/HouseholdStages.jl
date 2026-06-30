# Hall–Jones (2007) — health spending & the value of life

Health spending as a **share of income rises with income**, because the value of
life behaves as a luxury good: with curved (CRRA, σ>1) consumption utility the
marginal utility of consumption falls as people get richer, so trading consumption
for additional life-years — health/survival — becomes relatively more attractive.

This example is `examples/health` **extended** with a wealth/consumption margin and
a value-of-life flow, built entirely from existing stages.

## Household block (existing stages only)

Time order (`∘` runs the left stage first), auxiliary-choice-axis pattern:

    IncomeReceipt ∘ [ ChooseHealth' ∘ DebitHealth ∘ CommitHealth ∘ Forget ]
                  ∘ Consume ∘ Survive

- `IncomeReceipt`  — `WealthChangeStage(:wealth)`: cash-on-hand `(1+r)·wealth + w·y`.
- `ChooseHealth'`  — `ArgmaxStage` picks next health `h'` onto the auxiliary `:hc`
  axis (reward 0; the survival benefit enters via the continuation value, the cost
  is debited downstream — the `two_asset_hank` pattern).
- `DebitHealth`    — `WealthChangeStage(:wealth)` debits the medical cost of `h'`
  from wealth (reads `:hc`, `:health`, `:wealth`). **This couples health spending to
  the budget** so it competes with consumption — the precondition for a share result.
- `CommitHealth`   — `WealthChangeStage(:health)` writes `:health ← h'`.
- `Forget`         — `ForgetfulSumStage(:hc)` collapses the auxiliary axis.
- `Consume`        — `ConsumptionSavingsStage(:wealth)`, utility `u_crra(c) + vsl`
  (value-of-life flow while alive; zero when dead).
- `Survive`        — `MarkovStage(:alive | health)`: health-dependent survival on
  the chosen `h'`, dead absorbing.

No bespoke stage. The medical-cost debit and the value-of-life flow are economic
data fed to existing stages; the finite-horizon cohort sweep and the income sweep
are driver logic.

## Why ◐ (calibration, not structure)

The headline is a **calibration result living in the flow closures** — the
value-of-life flow `vsl`, the CRRA curvature `σ>1`, and a survival technology
`s(h) = s_max − gap·e^{−κh}` whose marginal benefit does **not** saturate over the
relevant range. With a saturating survival curve (or a health grid that pins at its
top), the share is hump-shaped, not monotone. The *structure* is just
`examples/health` + a consumption margin; the *result* is the calibration.

## Run

    julia --startup-file=no --project=. examples/hall_jones/steady_state.jl

### Output (confirms it solves + the headline result)

```
Hall–Jones value of life (N_age = 30, σ = 3.0, vsl = 4.00)
  V finite everywhere (all runs) = true
  cohort starts at mass 1; alive-mass decays along survival curve.
  income w   life exp     mean h(mid)  health share
  1.00       9.045        8.478        0.1627
  2.00       13.143       21.043       0.3786
  4.00       17.696       36.606       0.5017
  8.00       21.703       57.507       0.5461
  health-spending share rises with income = true (0.1627 → 0.5461)
```

The health-spending share of income rises monotonically with income (16% → 55%),
and life expectancy rises with income (9 → 22 ages) — the Hall–Jones value-of-life
mechanism. (Magnitudes are illustrative; they are a calibration of the flow
closures, not a structural prediction.)
