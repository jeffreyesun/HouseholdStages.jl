######################################################################
# Skill depreciation during unemployment — Ljungqvist–Sargent (1998)  #
######################################################################

# Search & savings with EMPLOYMENT-DEPENDENT human-capital dynamics: skill DECAYS
# while unemployed and GROWS/persists while employed. The decay raises the cost of a
# spell of unemployment (a low-skill worker who loses their job sees their re-employment
# wage fall as their skill erodes), which sharpens search incentives — the Ljungqvist–
# Sargent (1998 JPE) / Pissarides (1992 QJE) "loss of skill during unemployment"
# mechanism. The whole point of this Part-3 example: the within-period problem is FIVE
# existing library stages, in time order, with NO bespoke household stage —
#
#     Separation ∘ Matching ∘ SkillShock ∘ Receipt ∘ ConsumptionSavings
#
# (`∘` runs the LEFT stage first.)
#
# `Separation ∘ Matching` — ONE library call: `SearchMatchingStage` (derived sugar)
#               expands to `MarkovStage(separation) ∘ MixingStage(job-search
#               lottery)`; chains flatten, so the two leaves are unchanged.
#               `Separation` (on the two-level `:emp` axis, 1 = unemployed,
#               2 = employed): the employed lose their job w.p. `δ` (transition
#               `[1 0; δ 1−δ]`), BEFORE matching — a worker separated this period
#               searches this same period, so job loss and the search response are
#               not staggered across periods. `Matching`: the unemployed CHOOSE
#               their job-finding probability `p ∈ [0, 1]` directly — the lottery
#               over "search succeeds" (`[0 1; 0 1]`) and "search fails" (identity);
#               the employed rows coincide, so the employed choice is degenerate
#               (`p* = 0`, cost 0). The sugar single-homes the convex UTILS cost
#               `c(p) = κ_s·((1−p)log(1−p) + p)` and its closed-form argmax
#               `p*(y) = 1 − exp(−y/κ_s)`; the scale `κ_s = χ/(A_match·θ)` (with `θ`
#               read from `env`, the sugar's default) is calibrated so
#               `c′(p) = χ·e(p)` at the effort `e(p) = −log(1−p)/(A·θ)` the matching
#               technology `p = 1 − exp(−A·e·θ)` requires — higher tightness ⇒
#               cheaper search. The `:wealth`/`:skill` axes ride along as
#               spectators — a per-`(wealth, skill)` policy.
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
#   SearchMatchingStage (derived sugar = MarkovStage ∘ MixingStage), MarkovStage,
#   WealthChangeStage, ConsumptionSavingsStage.

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

    # Search-cost + matching primitives: κ_s(θ) = χ/(A_match·θ) scales the
    # probability-space search cost c(p) = κ_s·((1−p)log(1−p) + p).
    χ       :: Float64 = 0.5       # search-cost scale (effort disutility χ·e²/2)
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


# Household chain assembly — existing library stages only, NO bespoke stage #
#---------------------------------------------------------------------------#

"""
Build the skill-depreciation household block
`Separation ∘ Matching ∘ SkillShock ∘ Receipt ∘ ConsumptionSavings`, with employment
rate, mean wealth, and (employment-conditional) skill-sum moments attached. The `:emp`
axis is `[:unemp, :emp]` (level 1 = unemployed, 2 = employed). Separation + job search
is the `SearchMatchingStage` sugar — the separation `MarkovStage` composed with the
`MixingStage` lottery over the "success"/"fail" `:emp` kernels at convex
probability-space cost (tightness `θ` read from `env`, the sugar's default); the
`SkillShock` `MarkovStage`'s transition is the EMPLOYMENT-DEPENDENT dep closure
`(; emp) -> emp == :unemp ? T_decay : T_grow`. No bespoke household stage — the skill
kernels are economic data fed to existing stages.
"""
function skill_dep_household(p = skill_dep_params)
    n_skill = length(p.skill_grid)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :skill  => Discrete(p.skill_grid),
        :emp    => Discrete([:unemp, :emp]),       # 1 = unemployed, 2 = employed
    )

    search_and_match = SearchMatchingStage(layout;  # Separation ∘ Matching (derived sugar):
        separation          = p.δ,                  #   the separated search this same period;
        effort_cost_scale   = p.χ,                  #   tightness defaults to reading env.θ
        matching_efficiency = p.A_match,
    )

    T_decay = skill_drift_kernel(n_skill, p.p_decay; direction = :down)
    T_grow  = skill_drift_kernel(n_skill, p.p_grow;  direction = :up)
    skill_shock = MarkovStage(layout; axis = :skill,
        transition_matrix = (; emp) -> emp == :unemp ? T_decay : T_grow)

    receipt = WealthChangeStage(layout;            # defaults: (; axis = :wealth)
        wealth_post = (; wealth, skill, emp, env) -> (1 + env.r) * wealth +
                                     (emp == :emp ? env.w * skill : env.b_u),
    )

    savings = ConsumptionSavingsStage(layout;      # defaults: (; axis = :wealth)
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    )

    hh = search_and_match ∘ skill_shock ∘ receipt ∘ savings   # chains flatten: 5 leaves
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
