# Persistent + transitory income

The workhorse two-component earnings process (Storesletten-Telmer-Yaron
2004; Kaplan-Violante 2010): log income is the sum of a **persistent**
component and a **transitory** iid component,

    y_t = z_t · ν_t.

## The chain

    MarkovStage(:persistent) ∘ MarkovStage(:transitory) ∘ IncomeStage ∘ ConsumptionSavingsStage

Two **separate income axes** (`:persistent`, `:transitory`), each with its
own row-stochastic transition resolved by an independent `MarkovStage`. The
budget dep closure reads both:

    wealth_post = (; wealth, persistent, transitory, env) ->
        (1 + env.r) * wealth + env.w * persistent * transitory

The transitory chain is **iid** — `P_ν = repeat(p_ν', n)` gives identical
rows, the degenerate-Markov encoding of "next draw independent of current."

Built purely by composing existing exported stages; the second income axis
needs no new machinery, just a second `MarkovStage` and a budget closure
that names the extra axis.

## Run

    julia --startup-file=no --project=. examples/persistent_transitory/steady_state.jl

Fixed-r single solve (`r = 0.03 < 1/β − 1 ≈ 0.0417`).

## Result

    persistent grid : [0.6, 1.0, 1.4]
    transitory grid : [0.7, 1.0, 1.3]  (iid, row [0.25, 0.5, 0.25])
    ΣΛ      = 1.000000
    A_mean  = 3.65
