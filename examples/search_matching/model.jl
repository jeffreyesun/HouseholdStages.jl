####################################################################
# Labor search & matching with a tightness externality — household #
####################################################################

# A Mortensen–Pissarides / McCall income-fluctuation problem built ENTIRELY from
# existing HouseholdStages library stages — no bespoke household stage. The
# within-period problem decomposes into four stages, in time order:
#
#     Separation ∘ Matching ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# The `Separation ∘ Matching` pair ships as ONE library call: `SearchMatchingStage`
# (derived sugar) expands to exactly
# `MarkovStage(separation) ∘ MixingStage(job-search lottery)`;
# chains flatten, so the chain leaves are unchanged. The convex probability-space cost
# `c(p) = κ_s·((1−p)log(1−p) + p)`, its closed-form argmax `p*(y) = 1 − exp(−y/κ_s)`,
# and the calibration bridge `κ_s = χ/(A_match·θ)` are built into the sugar
# (single-homed in `src/stages/derived/search_matching.jl`) — no local cost/policy
# closures are rolled here.
#
# - `Separation` (`MarkovStage` on `:emp`): the employed lose their job w.p. `δ`
#   (transition `[1 0; δ 1−δ]`). TIMING: separation runs FIRST, so a worker
#   separated this period searches in the same period — job loss and the search
#   response are not staggered across periods.
# - `Matching` (`MixingStage` on `:emp`): the unemployed CHOOSE their job-finding
#   probability `p ∈ [0, 1]` directly — the lottery over `K_A` = "search succeeds"
#   (`[0 1; 0 1]`) and `K_B` = "search fails" (identity). The employed rows of the
#   two kernels coincide, so the employed choice is degenerate (`p* = 0`, cost 0).
#   The scale `κ_s = χ/(A_match·θ)` carries the tightness
#   externality: `c′(p) = χ·e(p)`, where `e(p) = −log(1−p)/(A·θ)` is the effort the
#   matching technology `p = 1 − exp(−A·e·θ)` requires to reach `p`, so the marginal
#   probability cost equals the marginal effort disutility `χ·e` at the implied
#   effort — higher tightness ⇒ cheaper search ⇒ higher employment.
#   The cost is a UTILITY cost (subtracted in the value recursion), as in McCall
#   search — not a resource drain on the budget. Tightness `θ` rides `env` (the
#   sugar's default) — exogenous in partial equilibrium, or closed by free entry
#   in the driver (`steady_state.jl`).
# - `IncomeReceipt` (`WealthChangeStage`): `b ↦ (1+r) b + income(emp)`, where the
#   employed earn wage `w` and the unemployed earn benefit `b_u`.
# - `ConsumptionSavingsStage`: pick `b_end` on the wealth grid; implicit budget
#   `c = b_in − b_end`; CRRA utility.
#
# The matching externality (the whole point): the job-finding cost every worker
# faces depends on aggregate tightness θ, which in GE is pinned by firms' free-entry
# vacancy posting (a purely outer-loop, firm-side condition — see steady_state.jl).
#
# Literature: McCall (1970) search; Mortensen & Pissarides (1994) matching;
# Pissarides (2000) "Equilibrium Unemployment Theory"; the income-fluctuation
# embedding follows Krusell, Mukoyama & Şahin (2010, ReStud) on search + savings.
#
# Library stages used (NO bespoke household stage):
#   SearchMatchingStage (derived sugar = MarkovStage ∘ MixingStage), WealthChangeStage,
#   ConsumptionSavingsStage.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct SearchMatchingParams
    β   :: Float64 = 0.96          # discount factor
    σ   :: Float64 = 2.0           # CRRA curvature
    r   :: Float64 = 0.03          # exogenous (PE) interest rate on savings
    w   :: Float64 = 1.0           # wage when employed
    b_u :: Float64 = 0.4           # unemployment benefit (income when unemployed)
    δ   :: Float64 = 0.10          # job-separation rate

    # Search-cost + matching primitives: κ_s(θ) = χ/(A_match·θ) scales the
    # probability-space search cost c(p) = κ_s·((1−p)log(1−p) + p).
    χ       :: Float64 = 0.5       # search-cost scale (effort disutility χ·e²/2)
    A_match :: Float64 = 0.5       # matching efficiency in p(e, θ) = 1 − exp(−A·e·θ)

    # Firm side (used only by the free-entry GE driver; pure outer-loop arithmetic).
    z     :: Float64 = 1.2         # match output (revenue per filled job)
    κ     :: Float64 = 0.30        # vacancy-posting cost
    η     :: Float64 = 0.5         # matching-function elasticity for q(θ) = A·θ^(−η)

    # Wealth grid.
    N_w   :: Int     = 120
    w_min :: Float64 = 0.0
    w_max :: Float64 = 40.0
end

Base.Broadcast.broadcastable(p::SearchMatchingParams) = Ref(p)

const search_matching_params = SearchMatchingParams()


# Utility: CRRA felicity `u_crra` is provided by HouseholdStages.


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached search-and-matching household block
`Separation ∘ Matching ∘ IncomeReceipt ∘ ConsumptionSavingsStage`, with the
employment rate and mean wealth attached as moments. The labor axis `:emp` has
level 1 = unemployed, 2 = employed; tightness `θ` is read from `env` (the
`SearchMatchingStage` sugar's default), so the same block serves PE at fixed `θ`
and the free-entry GE loop.
"""
function search_matching_household(p = search_matching_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :emp => Discrete([:unemp, :emp]),   # 1 = unemployed, 2 = employed
    )

    search_and_match = SearchMatchingStage(layout;   # Separation ∘ Matching (derived sugar):
        separation          = p.δ,                   #   the separated search this same period;
        effort_cost_scale   = p.χ,                   #   tightness defaults to reading env.θ
        matching_efficiency = p.A_match,
    )

    receipt = WealthChangeStage(layout;      # defaults: (; axis = :wealth)
        wealth_post = (; wealth, emp, env) -> (1 + env.r) * wealth +
                                     (emp == :emp ? env.w : env.b_u),
    )

    savings = ConsumptionSavingsStage(layout;   # defaults: (; axis = :wealth)
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    )

    hh = search_and_match ∘ receipt ∘ savings        # chains flatten: 4 leaves
    return define_moments!(hh;
        employment   = at_end(integrand = (; emp) -> emp == :emp ? 1.0 : 0.0,
                              reduce = sum),
        K_supplied   = at_end(integrand = :wealth, reduce = sum),
    )
end


# Firm side — free-entry tightness closure (plain outer-loop arithmetic) #
#-----------------------------------------------------------------------#

"""
Vacancy-filling rate `q(θ) = A·θ^(−η)` (clamped to `[0, 1]`), the firm-side leg of
the matching function. Falling in `θ`: a tighter market is harder to hire in.
"""
vacancy_fill(θ, p = search_matching_params) = clamp(p.A_match * θ^(-p.η), 0.0, 1.0)

"""
Value of a filled job to the firm, `J = (z − w) / (1 − β(1−δ))` — the discounted
flow surplus `z − w` over the match's expected life `1/(1−β(1−δ))`. A scalar
geometric sum (closed form), NOT a firm Bellman iteration: pure driver arithmetic.
"""
filled_job_value(p = search_matching_params) = (p.z - p.w) / (1 - p.β * (1 - p.δ))

"""
Free-entry residual `κ − q(θ)·J`: firms post vacancies until the cost `κ` equals the
expected hiring value `q(θ)·J`. Its root in `θ` is the GE tightness. Positive when
θ is too high (too few vacancies relative to free entry), negative when too low.
"""
free_entry_residual(θ, p = search_matching_params) =
    p.κ - vacancy_fill(θ, p) * filled_job_value(p)
