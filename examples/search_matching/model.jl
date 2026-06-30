####################################################################
# Labor search & matching with a tightness externality — household #
####################################################################

# A Mortensen–Pissarides / McCall income-fluctuation problem built ENTIRELY from
# existing HouseholdStages library stages — no bespoke household stage. The
# within-period problem decomposes into three stages, in time order:
#
#     SearchMatchingStage ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# - `SearchMatchingStage` (the dedicated effort-max + θ-dependent matching stage):
#   the unemployed choose search effort `e`; effort costs `cost(e)` utils and finds
#   a job with probability `job_finding(e, θ)` set by the matching function. The
#   employed separate at rate `δ`. Market tightness `θ` rides the `FromEnv(:θ)`
#   contract — exogenous in partial equilibrium, or closed by free entry in the
#   driver (`steady_state.jl`). The effort cost is a UTILITY cost (subtracted in the
#   value recursion), as in McCall search — it is not a resource drain on the budget.
# - `IncomeReceipt` (`WealthChangeStage`): `b ↦ (1+r) b + income(emp)`, where the
#   employed earn wage `w` and the unemployed earn benefit `b_u`.
# - `ConsumptionSavingsStage`: pick `b_end` on the wealth grid; implicit budget
#   `c = b_in − b_end`; CRRA utility.
#
# The matching externality (the whole point): the job-finding rate every worker
# faces depends on aggregate tightness θ, which in GE is pinned by firms' free-entry
# vacancy posting (a purely outer-loop, firm-side condition — see steady_state.jl).
#
# Literature: McCall (1970) search; Mortensen & Pissarides (1994) matching;
# Pissarides (2000) "Equilibrium Unemployment Theory"; the income-fluctuation
# embedding follows Krusell, Mukoyama & Şahin (2010, ReStud) on search + savings.
#
# Library stages used (NO bespoke household stage):
#   SearchMatchingStage, WealthChangeStage, ConsumptionSavingsStage.

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

    # Search effort + matching primitives.
    efforts :: Vector{Float64} = collect(range(0.0, 3.0; length = 12))
    χ       :: Float64 = 0.5       # effort-cost scale: cost(e) = χ·e²/2
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


# Search + matching primitives (plain functions) #
#------------------------------------------------#

"""
Search-effort utility cost `cost(e) = χ·e²/2` (a convex disutility, McCall-style).
"""
effort_cost(e, p = search_matching_params) = 0.5 * p.χ * e^2

"""
Job-finding probability `p(e, θ) = 1 − exp(−A·e·θ)`, increasing in own effort `e`
and in market tightness `θ`, bounded in `[0, 1)`. This is the channel through which
the tightness externality enters the worker's problem.
"""
job_finding(e, θ, p = search_matching_params) = 1 - exp(-p.A_match * e * θ)


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached search-and-matching household block
`SearchMatchingStage ∘ IncomeReceipt ∘ ConsumptionSavingsStage`, with the
employment rate and mean wealth attached as moments. The labor axis `:emp` has
level 1 = unemployed, 2 = employed; tightness `θ` is read from `env`.
"""
function search_matching_household(p = search_matching_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :emp => Discrete([:unemp, :emp]),   # 1 = unemployed, 2 = employed
    )

    matching = SearchMatchingStage(layout;   # defaults: (; axis = :emp, tightness = FromEnv(:θ))
        efforts     = p.efforts,
        cost        = e -> effort_cost(e, p),
        job_finding = (e, θ) -> job_finding(e, θ, p),
        separation  = p.δ,
    )

    receipt = WealthChangeStage(layout;      # defaults: (; axis = :wealth)
        wealth_post = (; wealth, emp, env) -> (1 + env.r) * wealth +
                                     (emp == :emp ? env.w : env.b_u),
    )

    savings = ConsumptionSavingsStage(layout;   # defaults: (; axis = :wealth, monotone_search = :divide_conquer)
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )

    hh = matching ∘ receipt ∘ savings
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
