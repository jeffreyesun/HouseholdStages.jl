######################################################################
# Rust (1987) optimal engine replacement — regenerative stopping      #
######################################################################

# Harold Zurcher's bus-engine problem (Rust, Econometrica 1987): the canonical
# regenerative optimal-stopping model. The single state is engine MILEAGE `x` on
# a grid. Each period the manager chooses `d ∈ {keep, replace}`:
#
#   KEEP    — pay the operating cost `c(x)` (rising in mileage); mileage then
#             deteriorates by a stochastic increment.
#   REPLACE — pay the replacement cost `RC` plus the operating cost of a fresh
#             engine `c(0)`; mileage RESETS to 0, then deteriorates from 0.
#
# Rust smooths the binary choice with i.i.d. Gumbel taste shocks of scale `ε`
# (the EV-logit). Writing the period boundary BEFORE the mileage shock — Rust's
# expected-value-function form `EV` — the integrated value is
#
#   EV(x⁻) = E[ W(x) | deteriorate from x⁻ ],
#     W(x)         = ε·log[ exp(v(x,keep)/ε) + exp(v(x,replace)/ε) ],
#     v(x,keep)    = −c(x)        + β·EV(x),
#     v(x,replace) = −(RC + c(0)) + β·EV(0).
#
# Here `x⁻` is the engine's mileage entering the period and `x` the mileage after
# deterioration (the mileage the manager observes and acts on). There is NO
# consumption/savings core: "value" is just the discounted stream of negative costs
# (a cost-minimisation Bellman). The point of THIS example is that the whole
# within-period problem is SIX existing library stages, in time order, with no
# bespoke stage — and that the regeneration (the reset-to-zero on replace) falls
# straight out of composition. Contrast `technology_adoption` (a network-externality
# adopt logit with NO regeneration); here the defining feature is the reset that
# makes the problem regenerative.
#
# Household block (time order = forward order, left → right):
#
#   Advance ∘ Choose ∘ FlowCost ∘ Discount ∘ Reset ∘ Forget
#
# `Advance`  — `MarkovStage` on `:mileage`: the stochastic deterioration increment.
#              It LEADS the block (built on the decision-singleton layout) so the
#              chain's boundary layout has `:decision` at size 1; the growing logit
#              cannot be the first stage, since the chain takes its input layout from
#              the first stage's construction layout (which the logit carries at the
#              full destination size 2).
# `Choose`   — `LogitChoiceStage` growing the transient `:decision` axis 1 → 2
#              ({keep, replace}). The cost matrix is all-zeros: every payoff is a
#              DESTINATION payoff (V-additive), so it lives in `FlowCost`, not in
#              the logit cost (the exact rule from logit_choice.jl / the exit
#              composite). `ε` is the Gumbel scale.
# `FlowCost` — `UtilityStage` adding the contemporaneous payoff per (mileage,
#              decision): `−c(x)` on keep, `−(RC + c(0))` on replace.
# `Discount` — `TimeDiscountingStage(β)` scaling ONLY the continuation (the flow
#              cost, added by the later-in-backward `FlowCost`, stays undiscounted).
# `Reset`    — `WealthChangeStage` on `:mileage` reading `:decision`: replace → 0,
#              keep → x. The regeneration, as a following stage (not baked into the
#              choice) — the same idiom as `BuyHomeStage ∘ WealthChangeStage`. Its
#              output is next period's boundary mileage `x⁻`.
# `Forget`   — `ForgetfulSumStage(:decision)` collapsing the transient axis 2 → 1.
#
# Backward (Forget → Reset → Discount → FlowCost → Choose → Advance) the Reset reads
# the boundary continuation `EV` at the per-branch post-mileage (`EV(0)` on replace,
# `EV(x)` on keep), Discount scales by β, FlowCost subtracts the period cost, the
# logit log-sum-exps the two branches into `W(x)`, and the Markov takes the
# deterioration expectation `EV(x⁻) = E[W(x)|x⁻]` — exactly the recursion above.
# Forward, deterioration moves each engine's mileage up, the logit SPLITS that mass
# across the two decisions by the softmax, Reset sends the replace branch to 0, and
# Forget sums the branches back: mass is conserved (no exit). Prices/costs are
# exogenous (partial equilibrium): no market to clear, so the outer loop is a single
# `solve_steady_state_given_env!`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct RustParams
    β    :: Float64 = 0.90                  # discount factor (Zurcher is patient; high β rewards replacement)
    ε    :: Float64 = 1.0                   # Gumbel (EV-logit) scale, in cost utils
    c1   :: Float64 = 1.0                   # operating-cost slope: c(x) = c1·x (rises with mileage)
    RC   :: Float64 = 150.0                 # lump replacement cost (utils); set so replacement concentrates high
                                            # (the keep/replace threshold lands at ~73% of the mileage grid)
    N_x   :: Int     = 61                   # mileage grid points
    x_max :: Float64 = 30.0                 # top of the mileage grid (engines are replaced before reaching it)
    g    :: Vector{Float64} = [0.05, 0.45, 0.50]  # deterioration: P(+0, +1, +2 bins); rows pile up at the top
end

Base.Broadcast.broadcastable(p::RustParams) = Ref(p)

const rust_params = RustParams()

"Operating / maintenance cost at mileage `x`: linear, rising in mileage; `c(0) = 0`."
opcost(x, p::RustParams) = p.c1 * x


# Deterioration transition (plain data handed to a MarkovStage) #
#---------------------------------------------------------------#

"""
The row-stochastic mileage-deterioration matrix `T[from, to]` for the `Advance` MarkovStage.
From bin `i`, mass moves up by `k−1` bins with probability `p.g[k]` (so `g[1]` = stay, `g[2]` = +1,
`g[3]` = +2 …); bins that would overshoot the top pile up at `N_x` (a reflecting cap, so engines that
are never replaced accumulate at maximum mileage). Each row sums to 1, so no mass leaks here.
"""
function deterioration_matrix(p::RustParams = rust_params)
    N = p.N_x
    T = zeros(Float64, N, N)
    for i in 1:N, (k, g) in enumerate(p.g)
        j = min(i + k - 1, N)
        T[i, j] += g
    end
    return T
end


# Household chain assembly #
#--------------------------#

"""
Build the Rust (1987) engine-replacement block `Choose ∘ FlowCost ∘ Discount ∘ Reset ∘ Advance ∘ Forget`
(six existing stages, no bespoke stage). The transient `:decision` axis is a singleton at the block
boundary and full (size 2) inside the choice block — the `LogitChoiceStage` grows it 1 → 2 and
`ForgetfulSumStage` collapses it back. `mean_mileage = ∫ x dΛ` is attached; the replacement rate is a
softmax-weighted moment the driver forms from `(V, Λ)` (the `:decision` axis is gone at block end).
"""
function rust_household(p = rust_params)
    xgrid = collect(range(0.0, p.x_max; length = p.N_x))
    mileage = GriddedContinuous(xgrid)
    # The `:decision` axis is transient: a SINGLETON at the period boundary (the block's input/output
    # layout, carried by the leading `Advance`) and FULL (1 = keep, 2 = replace) inside the choice block.
    # `Choose` grows 1 → 2; `Forget` collapses 2 → 1 — the auxiliary-choice-axis pattern, exactly as the
    # exit composite threads `:exiting`. The growing logit must NOT lead the chain (the chain inherits its
    # boundary layout from the first stage's construction layout), so `Advance` leads on `block`.
    block = GriddedLayout(:mileage => mileage, :decision => Discrete([1]))
    full  = GriddedLayout(:mileage => mileage, :decision => Discrete([1, 2]))   # 1 = keep, 2 = replace

    # Advance: stochastic deterioration of mileage at the start of the period (decision axis still 1).
    advance = MarkovStage(block; axis = :mileage, transition_matrix = deterioration_matrix(p))

    # Choose: smooth keep/replace. The cost matrix is all-zeros (origin singleton → 2 destinations); the
    # real payoffs are destination payoffs and so are V-additive — they belong in `FlowCost`, never here.
    choose = LogitChoiceStage(full; axis = :decision, cost_matrix = zeros(1, 2), ε = p.ε)

    # FlowCost: the contemporaneous cost. KEEP pays operating cost c(x); REPLACE pays RC + c(0).
    flowcost = UtilityStage(full;
        utility = (; mileage, decision, env) ->
            decision == 1 ? -opcost(mileage, p) : -(env.RC + opcost(0.0, p)))

    # Discount: scale the continuation only (the flow cost is added later in backward, undiscounted).
    discount = TimeDiscountingStage(full; β = p.β)

    # Reset: the regeneration. REPLACE sends mileage to 0; KEEP leaves it at the realized mileage x.
    reset = WealthChangeStage(full; axis = :mileage,
        wealth_post = (; mileage, decision) -> decision == 2 ? 0.0 : mileage)

    # Forget: collapse the transient decision axis, summing the keep and replace branches.
    forget = ForgetfulSumStage(full; axis = :decision)

    hh = advance ∘ choose ∘ flowcost ∘ discount ∘ reset ∘ forget
    return define_moments!(hh;
        mean_mileage = at_end(integrand = :mileage, reduce = sum))   # ∫ x dΛ over the post-reset boundary
end


# Exogenous environment (plain function, no AbstractBlock) #
#----------------------------------------------------------#

"The env consumed by the chain: the lump replacement cost (the one cost read from env)."
rust_env(p = rust_params) = (; RC = p.RC)
