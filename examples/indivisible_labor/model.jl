#######################################################################
# Indivisible labor (Rogerson 1988) — extensive-margin participation   #
#######################################################################

# Labor is INDIVISIBLE: a household either works full-time hours `n̄` or not at all
# (a discrete {work, not-work} participation choice), then chooses savings. Working
# earns `w·ε·n̄` and costs a fixed disutility `v(n̄)`; not working earns nothing and
# costs nothing. References: Rogerson (1988), Prescott–Rogerson–Wallenius (2009),
# Chang–Kim (2007, extensive margin).
#
# THE ✅ BUILD — discrete participation as an ArgmaxStage (sub-approach (a)). The
# participation choice is a `(max, +)` over a 2-level `:participation` axis whose reward
# encodes the work disutility; the chosen participation feeds the budget (the
# auxiliary-axis pattern, as in two_asset_hank). Household block, existing stages only:
#
#   EmploymentShock ∘ Participate ∘ Budget(reads participation) ∘ ConsumptionSavings
#
# `Participate` (ArgmaxStage on :participation) — picks work (n̄) vs not, paying the
#   reward `−v(n̄)·1{work}`. The reward is INDEPENDENT of the incoming participation
#   state, so every cell re-selects the same optimal participation regardless of last
#   period's choice — i.e. the choice is genuinely WITHIN-PERIOD even though the axis
#   persists across periods (the persistence is harmless and lets us read the chosen
#   participation as an end-of-period moment).
# `Budget` (WealthChangeStage on :wealth) — cash-on-hand `(1+r)b + w·ε·n̄·1{work}`,
#   reading the chosen `:participation` indicator.
# `ConsumptionSavings` (ConsumptionSavingsStage on :wealth) — CRRA `u(c)`; the work
#   disutility already sits in the participation reward (separable preferences
#   `u(c) − v(n)`).
#
# Period value: V_start = max_p [ −v(n̄)·1{p=work} + max_{b'} ( u(c_p) + β·E V' ) ],
# with c_p = (1+r)b + w·ε·n̄·1{p=work} − b'. The outer max over p is the ArgmaxStage;
# its V_end is the inner consumption-savings continuation per participation. Whether an
# agent works trades the labor income (a higher continuation through Budget) against the
# disutility −v(n̄): productive (high-ε) and asset-poor agents work; the income effect
# (low marginal utility of consumption when rich) pulls wealthy agents OUT — the
# reservation-wealth property of the extensive margin.
#
# THE ROGERSON LOTTERY (sub-approach (b)) — NOT used; see README for the precise reason.
# The convexifying employment lottery is NOT faithfully a `MixingStage`: MixingStage
# blends two TRANSITIONS on a single axis at a CONVEX cost `c(θ)`, but the work/not
# difference lives in the BUDGET (a wealth-axis move) and a flat reward, not in a
# participation-axis transition, and the Rogerson lottery is valued LINEARLY in θ (no
# convex cost) so its individual problem is exactly the corner discrete choice (a). The
# discrete ArgmaxStage build IS the faithful incomplete-markets indivisible-labor model
# (Chang–Kim): participation is a cross-sectional rate, not an individual lottery.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct IndivisibleLaborParams
    β :: Float64       = 0.96
    σ :: Float64       = 2.0      # CRRA over consumption
    r :: Float64       = 0.03     # FIXED exogenous return, < 1/β − 1 ≈ 0.0417 (partial equilibrium)
    w :: Float64       = 1.0      # wage per efficiency unit
    nbar :: Float64    = 1.0      # indivisible full-time hours
    # Fixed disutility of working v(n̄). Calibrated to deliver an interior participation
    # rate; equals ψ·n̄^{1+1/φ}/(1+1/φ) in the separable-CRRA parameterization.
    disutil :: Float64 = 1.1
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]   # labor-efficiency (productivity) states ε
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    # Participation indicator axis: 0 = not work, 1 = work.
    part_grid :: Vector{Float64} = [0.0, 1.0]
    N_w   :: Int       = 150
    w_min :: Float64   = 0.0
    w_max :: Float64   = 40.0
end

Base.Broadcast.broadcastable(p::IndivisibleLaborParams) = Ref(p)

const indivisible_labor_params = IndivisibleLaborParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached indivisible-labor household block
`EmploymentShock ∘ Participate ∘ Budget ∘ ConsumptionSavings`. `Participate` is a square
`ArgmaxStage` on the 2-level `:participation` axis whose reward `−v(n̄)·1{work}` is
independent of the incoming participation, so the choice is re-made within-period; `Budget`
reads the chosen indicator to set cash-on-hand `(1+r)b + w·ε·n̄·1{work}`; the consumption
disutility of work lives in the participation reward (separable preferences). Attaches
`mean_wealth_agg = ∫ wealth` and `participation_agg = ∫ 1{work}` (divide each by mass).
"""
function indivisible_labor_household(p = indivisible_labor_params)
    layout = GriddedLayout(
        :wealth        => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income        => Discrete(p.y_grid),
        :participation => Discrete(p.part_grid),
    )

    # Square 2×2 participation reward, columns (the incoming state) identical: choosing
    # `work` (row 2) pays −v(n̄); `not` (row 1) pays 0. `reward[after, before]`.
    vbar          = p.disutil
    part_reward   = [0.0   0.0;
                     -vbar -vbar]

    shock    = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    choose   = ArgmaxStage(layout; axis = :participation, reward = part_reward)
    budget   = WealthChangeStage(layout; axis = :wealth,                       # cash = (1+r)b + w·ε·n̄·1{work}
        wealth_post = (; wealth, income, participation, env) ->
            (1 + env.r) * wealth + env.w * income * p.nbar * participation)
    savings  = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth)

    hh = shock ∘ choose ∘ budget ∘ savings
    return define_moments!(hh;
        mean_wealth_agg   = at_end(integrand = :wealth, reduce = sum),
        participation_agg = at_end(integrand = (; participation) -> participation, reduce = sum),
    )
end

"""
Env for the indivisible-labor partial-equilibrium experiment: the fixed return `r` and
wage `w` consumed by the budget closure.
"""
indivisible_labor_env(p = indivisible_labor_params) = (; r = p.r, w = p.w)
