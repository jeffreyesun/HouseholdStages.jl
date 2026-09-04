# Ilut–Schneider (2014) — ambiguous business cycles

**Ilut & Schneider (2014, AER), "Ambiguous Business Cycles."** EXAMPLES.md §2 (robustness & ambiguity). Status: ◐ — the **entropic/smoothed form is built here ✅**; the literal set-based max-min is its θ → 0⁺ limit.

## Mechanism

The household is **ambiguity-averse about the aggregate business cycle**. It does not trust the aggregate transition `P_z` over the persistent cycle state `z ∈ {boom, recession}`; it acts on a **worst-case** mean from an entropy-constrained ambiguity set around `P_z` (max-min utility). The smoothed/entropic version of that worst-case operator is

```
V_start[z] = −θ · log Σ_z′ P_z[z,z′] · exp(−V_end[z′]/θ),
```

with the literal set-based max-min recovered as θ → 0⁺ (ε → 0⁻). Idiosyncratic income risk is **cycle-dependent** (recessions sticky-low), so pessimism about the cycle is pessimism about one's own future income — the Ilut–Schneider amplification channel.

This is built as a sibling of `examples/regime_switching` (same aggregate-cycle + cycle-dependent-income structure), and is **deliberately differentiated from `examples/hansen_sargent`**: there the ε<0 tilt acts on the *idiosyncratic* income transition; here it acts on the *persistent aggregate* cycle transition, while idiosyncratic income is an ordinary trusted `MarkovStage` selected by the cycle.

## Exact chain (existing stages only)

```
AmbiguityTilt ∘ IncomeDraw ∘ Receipt ∘ ConsumptionSavings
   = LogitChoiceStage(:z, ε = −θ, cost = −ε·log P_z)
        ∘ MarkovStage(:income; transition = (; z) -> T_income(z))
        ∘ IncomeStage
        ∘ ConsumptionSavingsStage(β, u_crra)
```

The worst-case cycle operator **is** `LogitChoiceStage` on the `:z` axis at **negative ε**, with `cost = −ε·log P_z` so that `exp(−C/ε) = P_z` (ε cancels) and the logit backward becomes exactly the entropic worst-case `−θ·log Σ_z′ P_z[z,z′]·exp(−V_end[z′]/θ)`, multiplier `|ε| = θ`. The cycle-dependent idiosyncratic income is a `(; z) ->` dep-closure handed to an ordinary `MarkovStage`. No bespoke stage, kernel, or `forward!`/`backward!` was written.

`P_z` is kept strictly positive (no `log 0`).

## Fidelity

**Faithful** for the entropic (smoothed) form — the worst-case value recursion is reproduced exactly, and the θ → ∞ limit matches the ordinary-expectation `MarkovStage(:z)` reference chain (mean wealth and recession share) to ~1e-8. The literal set-based max-min (◐) is the θ → 0⁺ limit, not separately implemented.

Unlike `examples/hansen_sargent`, the worst-case forward measure is **economically desirable to display here**: `LogitChoiceStage.forward!` pushes Λ through the seated pessimistic cycle kernel, so the ergodic cycle distribution tilts toward recessions — the "ambiguous business cycles" amplification. We report this directly: the worst-case recession share rises far above the reference ergodic share as ambiguity increases.

## Run

```
julia --startup-file=no --project=. examples/ilut_schneider/steady_state.jl
```

Headline numbers (default calibration, N_w = 120):
- **Precaution:** `E_ref[b']` (chosen next-wealth under a common reference distribution) rises from 1.166 (θ = 1e8) to 1.283 (θ = 0.5), **+10.1%**, monotone in θ.
- **Amplification:** worst-case recession share rises from 0.286 (reference ergodic) to **0.921** at θ = 0.5, monotone in θ.
- **θ → ∞ cross-check:** mean wealth 2.0787 vs reference 2.0787 (|Δ| = 1.2e-8); recession share 0.2857 vs 0.2857 (|Δ| = 2.9e-9).
