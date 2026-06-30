######################################################################
# Skill depreciation during unemployment — Ljungqvist–Sargent (1998)  #
######################################################################

# Search & savings with EMPLOYMENT-DEPENDENT human-capital dynamics: skill DECAYS
# while unemployed and GROWS/persists while employed. The decay raises the cost of a
# spell of unemployment (a low-skill worker who loses their job sees their re-employment
# wage fall as their skill erodes), which sharpens search incentives — the Ljungqvist–
# Sargent (1998 JPE) / Pissarides (1992 QJE) "loss of skill during unemployment"
# mechanism. The whole point of this Part-3 example: the within-period problem is FOUR
# existing library stages, in time order, with NO bespoke household stage —
#
#     Matching ∘ SkillShock ∘ Receipt ∘ ConsumptionSavings
#
# (`∘` runs the LEFT stage first.)
#
# `Matching`  — `SearchMatchingStage` on the two-level `:emp` axis (1 = unemployed,
#               2 = employed). The unemployed choose search effort `e`; effort costs
#               `cost(e)` utils and finds a job with probability `job_finding(e, θ)`;
#               the employed separate at rate `δ`. Tightness `θ` is a fixed scalar
#               (partial equilibrium). The `:wealth`/`:skill` axes ride along as
#               spectators — the stage solves a per-`(wealth, skill)` effort policy.
#
# `SkillShock` — `MarkovStage` on the `:skill` axis whose ROW-stochastic transition is
#               EMPLOYMENT-DEPENDENT via a dep closure `(; emp) -> emp == :unemp ?
#               T_decay : T_grow` (exactly the health.jl `(; health) -> T` pattern, keyed
#               on `:emp`). `T_decay` drifts skill DOWN one rung w.p. `p_decay` (the
#               erosion of human capital in a jobless spell); `T_grow` drifts it UP one
#               rung w.p. `p_grow` (learning-by-doing on the job). Both are banded,
#               row-stochastic, with reflecting ends. Plain economic data fed to the
#               EXISTING `MarkovStage`, not a new stage.
#
# `Receipt`   — `WealthChangeStage` `b ↦ (1+r) b + income`, where the employed earn
#               `w · skill` (the wage scales with human capital — this is what skill
#               decay erodes) and the unemployed earn benefit `b_u`.
#
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period wealth `b'`;
#               implicit budget `c = b_in − b'`; CRRA utility.
#
# Partial equilibrium: tightness `θ`, return `r`, wage scale `w`, benefit `b_u` are all
# exogenous, so the "outer loop" is a single `solve_steady_state_given_env!`. The decay
# mechanism shows up in the stationary distribution as LOWER mean skill among the
# unemployed than the employed.
#
# Literature: Ljungqvist & Sargent (1998 JPE) skill-loss & the European unemployment
# experience; Pissarides (1992 QJE) loss of skill & persistence of unemployment; the
# search+savings embedding follows Krusell, Mukoyama & Şahin (2010 ReStud).
#
# Library stages used (NO bespoke household stage):
#   SearchMatchingStage, MarkovStage, WealthChangeStage, ConsumptionSavingsStage.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct SkillDepParams
    β   :: Float64 = 0.96          # discount factor
    σ   :: Float64 = 2.0           # CRRA curvature
    r   :: Float64 = 0.03          # exogenous (PE) interest rate on savings
    w   :: Float64 = 1.0           # wage scale when employed (income = w·skill)
    b_u :: Float64 = 0.4           # unemployment benefit (income when unemployed)
    δ   :: Float64 = 0.10          # job-separation rate
    θ   :: Float64 = 1.0           # market tightness (fixed scalar, partial equilibrium)

    # Search effort + matching primitives.
    efforts :: Vector{Float64} = collect(range(0.0, 3.0; length = 10))
    χ       :: Float64 = 0.5       # effort-cost scale: cost(e) = χ·e²/2
    A_match :: Float64 = 0.5       # matching efficiency in p(e, θ) = 1 − exp(−A·e·θ)

    # Skill ladder: a small grid of human-capital levels (wage multipliers).
    skill_grid :: Vector{Float64} = [0.6, 0.8, 1.0, 1.2, 1.4]
    p_decay    :: Float64 = 0.40   # P(skill drops one rung | unemployed)
    p_grow     :: Float64 = 0.25   # P(skill rises one rung | employed)

    # Wealth grid.
    N_w   :: Int     = 100
    w_min :: Float64 = 0.0
    w_max :: Float64 = 40.0
end

Base.Broadcast.broadcastable(p::SkillDepParams) = Ref(p)

const skill_dep_params = SkillDepParams()


# Utility: CRRA felicity `u_crra` is provided by HouseholdStages.


# Search-effort primitives (plain functions) #
#--------------------------------------------#

"""
Search-effort utility cost `cost(e) = χ·e²/2` (a convex disutility, McCall-style).
"""
effort_cost(e, p = skill_dep_params) = 0.5 * p.χ * e^2

"""
Job-finding probability `p(e, θ) = 1 − exp(−A·e·θ)`, increasing in own effort `e`
and in market tightness `θ`, bounded in `[0, 1)`.
"""
job_finding(e, θ, p = skill_dep_params) = 1 - exp(-p.A_match * e * θ)


# Skill-transition kernels (plain economic helpers — row-stochastic on the skill grid) #
#--------------------------------------------------------------------------------------#

"""
Build a banded, row-stochastic skill-drift kernel `T[from, to]` on an `n`-rung ladder.
With `direction = :down`, each rung moves one step DOWN w.p. `p` and stays w.p. `1−p`
(the bottom rung is reflecting — it stays w.p. 1); with `direction = :up`, one step UP
w.p. `p` (the top rung reflecting). This is the discrete analogue of a one-sided drift
on the human-capital ladder — plain data handed to `MarkovStage`, not a new stage.
"""
function skill_drift_kernel(n::Integer, p::Real; direction::Symbol)
    T = zeros(Float64, n, n)
    for i in 1:n
        if direction === :down
            if i == 1
                T[i, i] = 1.0                       # bottom rung reflects
            else
                T[i, i - 1] = p
                T[i, i]     = 1 - p
            end
        elseif direction === :up
            if i == n
                T[i, i] = 1.0                       # top rung reflects
            else
                T[i, i + 1] = p
                T[i, i]     = 1 - p
            end
        else
            error("skill_drift_kernel: direction must be :down or :up, got $direction")
        end
    end
    return T
end


# Household chain assembly — FOUR library stages, NO bespoke stage #
#-----------------------------------------------------------------#

"""
Build the skill-depreciation household block
`Matching ∘ SkillShock ∘ Receipt ∘ ConsumptionSavings`, with employment rate, mean
wealth, and (employment-conditional) skill-sum moments attached. The `:emp` axis is
`[:unemp, :emp]` (level 1 = unemployed, 2 = employed). The `SkillShock` `MarkovStage`'s
transition is the EMPLOYMENT-DEPENDENT dep closure `(; emp) -> emp == :unemp ? T_decay :
T_grow`; tightness `θ` is read from `env`. No bespoke household stage — the skill kernels
are economic data fed to `MarkovStage`.
"""
function skill_dep_household(p = skill_dep_params)
    n_skill = length(p.skill_grid)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :skill  => Discrete(p.skill_grid),
        :emp    => Discrete([:unemp, :emp]),       # 1 = unemployed, 2 = employed
    )

    matching = SearchMatchingStage(layout;         # defaults: (; axis = :emp, tightness = FromEnv(:θ))
        efforts     = p.efforts,
        cost        = e -> effort_cost(e, p),
        job_finding = (e, θ) -> job_finding(e, θ, p),
        separation  = p.δ,
    )

    T_decay = skill_drift_kernel(n_skill, p.p_decay; direction = :down)
    T_grow  = skill_drift_kernel(n_skill, p.p_grow;  direction = :up)
    skill_shock = MarkovStage(layout; axis = :skill,
        transition_matrix = (; emp) -> emp == :unemp ? T_decay : T_grow)

    receipt = WealthChangeStage(layout;            # defaults: (; axis = :wealth)
        wealth_post = (; wealth, skill, emp, env) -> (1 + env.r) * wealth +
                                     (emp == :emp ? env.w * skill : env.b_u),
    )

    savings = ConsumptionSavingsStage(layout;      # defaults: (; axis = :wealth, monotone_search = :divide_conquer)
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )

    hh = matching ∘ skill_shock ∘ receipt ∘ savings
    return define_moments!(hh;
        employment      = at_end(integrand = (; emp) -> emp == :emp ? 1.0 : 0.0,           reduce = sum),
        unemployment    = at_end(integrand = (; emp) -> emp == :unemp ? 1.0 : 0.0,         reduce = sum),
        mean_wealth     = at_end(integrand = :wealth,                                      reduce = sum),
        skill_emp_sum   = at_end(integrand = (; emp, skill) -> emp == :emp ? skill : 0.0,  reduce = sum),
        skill_unemp_sum = at_end(integrand = (; emp, skill) -> emp == :unemp ? skill : 0.0, reduce = sum),
    )
end


# Exogenous env (plain function, partial equilibrium) #
#-----------------------------------------------------#

"The exogenous price/tightness env for the skill-depreciation household."
skill_dep_env(p = skill_dep_params) = (; θ = p.θ, r = p.r, w = p.w, b_u = p.b_u)
