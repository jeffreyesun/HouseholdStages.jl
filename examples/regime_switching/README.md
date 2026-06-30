# Regime-switching income (Hamilton 1989-style)

Aggregate conditions follow a Markov **regime** (boom / recession). The
idiosyncratic income transition depends on the current regime: in a
recession the conditional income distribution is shifted toward, and
stickier in, the low state.

Unlike `countercyclical_risk` (where the regime is an exogenous `env` input
fixed within a steady state), here the regime is an **endogenous axis** with
its own Markov law, so one stationary Λ carries the joint
`(regime, income, wealth)` distribution and both aggregate states coexist.

## The chain

    MarkovStage(:regime)
        ∘ MarkovStage(:income; transition_matrix = (; regime) -> T_income(regime))
        ∘ IncomeStage ∘ ConsumptionSavingsStage

The income transition is a **dep-closure on the `:regime` axis**.
`MarkovStage` reads the closure's `regime` kwarg via `Base.kwarg_decl`,
recognizes it as a layout axis, and stores one income transition per regime
value (a compact `(n_income, n_income, n_regime)` field); each regime cell
picks its own income fiber at apply time. The closure receives the regime
**grid value** (not an index), so it dispatches on `regime == REGIME_BOOM`.

This is the spec's "if the dep-varying transition can't be made to work,
report what failed" case — **it works**, with no new machinery.

## Run

    julia --startup-file=no --project=. examples/regime_switching/steady_state.jl

Fixed-r single solve.

## Result

    ΣΛ              = 1.000000
    A_mean          = 2.10
    recession_share = 0.2000   (regime-chain ergodic = 0.2000)

The recession share recovered as a moment of the joint Λ exactly matches the
regime chain's own ergodic mass — a cross-check that the endogenous regime
axis evolved correctly and the regime-dependent income transition seated per
regime.
