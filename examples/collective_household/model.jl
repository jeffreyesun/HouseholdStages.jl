###############################################################
# Collective Household — fixed Pareto weight                   #
# (Chiappori 1988/1992; Mazzocco 2007)                         #
###############################################################
#
# A heterogeneous-agent consumption-savings model in which the "agent" is a
# two-member household that allocates a SINGLE shared budget between its
# members. Under the collective approach, an efficient household maximises a
# Pareto-weighted sum of member utilities. With a FIXED Pareto weight `μ`
# the household is observationally a single planner — it solves
#
#     max  μ·u_A(c_A) + (1−μ)·u_B(c_B)   s.t.   one shared budget,
#
# and a sharing rule `c_A = s·c`, `c_B = (1−s)·c` splits the common
# consumption `c`. A fixed weight makes the per-period objective a reshaped
# felicity over the single choice `c`, so the whole model is the Aiyagari
# spine with that felicity.
#
# The within-period BLOCK is the canonical Aiyagari spine — three EXISTING
# exported stages, no bespoke stage:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#     = MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage
#
# The ONLY thing that makes this a collective household is the felicity
# closure handed to `ConsumptionSavingsStage`:
#
#     μ·u_crra(s·c, σ_A) + (1−μ)·u_crra((1−s)·c, σ_B)
#
# Members differ in CRRA curvature (`σ_A ≠ σ_B`), so even with a fixed weight
# the household's effective curvature is a member-weighted blend — the
# collective content. `u_crra` masks `c ≤ 0` (hence both member shares ≤ 0)
# to `-Inf`. No extra `utility_axes` are needed.
#
# We deliberately build ONLY the fixed-weight (full-commitment) version. The
# evolving-weight / limited-commitment collective model (Mazzocco 2007;
# Voena 2015; Alvarez–Jermann-style participation constraints) makes the
# Pareto weight a state that updates with binding participation constraints,
# which is NOT a felicity reshape of a single planner and is out of scope here.
#
# Partial equilibrium: `r`, `w` fixed and exogenous; one inner solve.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CollectiveParams
    β   :: Float64 = 0.96
    σ_A :: Float64 = 1.5                 # CRRA curvature of member A
    σ_B :: Float64 = 3.0                 # CRRA curvature of member B (more risk-averse)
    μ   :: Float64 = 0.6                 # FIXED Pareto weight on member A
    s   :: Float64 = 0.55                # sharing rule: c_A = s·c, c_B = (1−s)·c
    r   :: Float64 = 0.02                # FIXED real return, < 1/β − 1 ≈ 0.0417
    w   :: Float64 = 1.0                 # FIXED wage (scales the endowment)
    # Three-state idiosyncratic income process (persistent, mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 1.0, 1.5]
    P_y    :: Matrix{Float64} = [0.75 0.20 0.05;
                                 0.15 0.70 0.15;
                                 0.05 0.20 0.75]
    N_w   :: Int     = 150               # wealth grid points
    w_min :: Float64 = 0.0
    w_max :: Float64 = 100.0
end

Base.Broadcast.broadcastable(p::CollectiveParams) = Ref(p)

const collective_params = CollectiveParams()


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached collective-household block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`. The savings closure
splits the chosen common consumption `c` by the sharing rule and evaluates the
fixed-Pareto-weight objective `μ·u_A(s·c) + (1−μ)·u_B((1−s)·c)`, turning the
Aiyagari spine into a single-planner collective household. Attaches the
aggregate-wealth moment `K_supplied = ∫ wealth dΛ`.
"""
function collective_household(p = collective_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout)            # (1+r)·b + w·y
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        # Collective felicity: Pareto-weighted sum of the two members'
        # utilities over their shares of the common consumption c.
        utility = (cell, c; env) ->
            p.μ * u_crra(p.s * c, Val(p.σ_A)) +
            (1 - p.μ) * u_crra((1 - p.s) * c, Val(p.σ_B)),
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Env (plain function, no AbstractBlock) #
#----------------------------------------#

"""
Env for the partial-equilibrium collective-household experiment: fixed real
return `r` and wage `w`. One inner solve delivers the stationary distribution.
"""
collective_env(p = collective_params) = (; r = p.r, w = p.w)
