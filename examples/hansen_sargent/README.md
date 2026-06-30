# Hansen–Sargent multiplier preferences / robust savings

**Hansen–Sargent (2008) multiplier preferences; Hansen–Sargent–Tallarini (1999) risk-sensitive control.** MODEL_CATALOG.md §2 (robustness) and §7 (the headline negative-ε flip). Status: ✅.

## Mechanism

A robust incomplete-markets saver does not trust its income transition `P_y`. Rather than take the ordinary expectation of next-period continuation, it **minimises over entropy-penalised distortions** of `P_y`, paying `θ · KL(distorted ‖ P_y)` per unit of relative entropy. The minimised object is the exponential/entropic certainty equivalent

```
V_start[i] = −θ · log Σ_j P_y[i,j] · exp(−V_end[j]/θ),     θ = robustness multiplier > 0.
```

Small θ ⇒ very robust / pessimistic; θ → ∞ ⇒ ordinary expectation (risk-neutral continuation).

## Exact chain (existing stages only)

```
RobustExpectation ∘ Receipt ∘ ConsumptionSavings
   = LogitChoiceStage(:income, ε = −θ, cost = −ε·log P_y)
        ∘ IncomeStage
        ∘ ConsumptionSavingsStage(β, u_crra)
```

The risk-sensitive operator **is** `LogitChoiceStage` at **negative ε**. The logit backward computes `V_start[i] = ε·log Σ_j exp((−C[i,j]+V_end[j])/ε)`. Choosing `C[i,j] = −ε·log P_y[i,j]` makes `exp(−C/ε) = P_y` (the ε cancels exactly), so the recursion collapses to `ε·log Σ_j P_y[i,j]·exp(V_end[j]/ε) = −θ·log Σ_j P_y[i,j]·exp(−V_end[j]/θ)` with `ε = −θ` — the entropic certainty equivalent, multiplier `|ε| = θ`. No bespoke stage, kernel, or `forward!`/`backward!` was written: robustness is the soft-MIN member of the same log-sum-exp that gives logit discrete choice at `ε > 0` (§7, the choice↔robustness flip).

`P_y` is kept strictly positive (no `log 0`).

## Fidelity

**Faithful** for the value recursion / savings policy — the entropic CE is reproduced exactly, and the θ → ∞ limit matches the ordinary-expectation `MarkovStage` chain to ~1e-7 (numerical cross-check in `steady_state.jl`).

One genuine framework characteristic to flag: `LogitChoiceStage.forward!` pushes the wealth distribution through the **seated worst-case kernel** `π(j|i) ∝ P_y[i,j]·exp(−V_end[j]/θ)`, so the stationary Λ is the ergodic distribution under the household's *pessimistic* belief — value-distortion and simulated measure are coupled in one stage. The cautious policy and the pessimistic simulated income then push own-measure mean wealth in opposite directions, so that statistic is **non-monotone** in θ and is *not* a clean precaution read. (A robust-control exercise that wants the value under the worst-case measure but the distribution under the reference measure `P_y` would need two distinct measures in backward vs forward — the single stage does not separate them.)

The precautionary effect is therefore demonstrated **distribution-free**, via the savings policy: the average chosen next-period wealth `E_ref[b'(state)]` under a common fixed reference distribution is cleanly **monotone decreasing in θ** (more robust ⇒ carries more wealth), and mean V is monotone (pessimism lowers the value). This is the standard and unambiguous reading of the precautionary tilt.

## Run

```
julia --startup-file=no --project=. examples/hansen_sargent/steady_state.jl
```

Headline numbers (default calibration, N_w = 120): `E_ref[b']` rises from 1.931 (θ = 1e8, ≈ risk-neutral) to 2.235 (θ = 0.5, robust), **+15.7%**, monotone in θ; θ = 1e8 own-measure mean wealth = 2.8753 vs `MarkovStage` reference 2.8753 (|Δ| = 8.3e-8).
