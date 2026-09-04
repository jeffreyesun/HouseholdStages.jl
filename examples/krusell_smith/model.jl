################################################################
# Krusell-Smith (1998) — Deterministic-Aggregate Steady State #
################################################################

# Same within-period structure as Aiyagari (`../aiyagari/model.jl`),
# specialised to the K-S employed/unemployed income process. The
# household problem decomposes into three stages, in time order:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# `IncomeShock` resolves the two-state Markov draw on the income axis
# (employed `y = 1`, unemployed `y = 0.07` — the canonical K-S
# calibration). `IncomeReceipt` is the deterministic wealth update
# `b ↦ (1+r) b + w y`. `ConsumptionSavingsStage` then chooses
# next-period wealth on the wealth grid with implicit budget
# `c = b_in - b_end`. Production is Cobb-Douglas with the TFP level
# `A` carried explicitly so a deterministic-aggregate driver
# (`steady_state.jl`) can sweep it; effective labor is the stationary
# distribution of the income chain.
#
# The wealth grid is log-spaced for the same reason as in Aiyagari.
# K-S calibration runs at a higher capital level (K = 12.88 vs
# Aiyagari's K = 5.69), so `w_max` is scaled up to 200 to keep the
# active region well-resolved.

using HouseholdStages
using LinearAlgebra: I


# Parameters #
#------------#

@kwdef struct KSParams
    β :: Float64       = 0.96
    γ :: Float64       = 1.0

    α :: Float64       = 0.36
    δ :: Float64       = 0.025

    # Idiosyncratic productivity: unemployed / employed. `y_unemp = 0.07`
    # is the canonical K-S calibration (Krusell & Smith 1998); a strictly
    # positive unemployed income is needed under log utility so the
    # b = 0 corner remains feasible.
    y_grid :: Vector{Float64} = [0.07, 1.0]
    P_y    :: Matrix{Float64} = [0.6   0.4;
                                 0.05  0.95]

    N_w   :: Int       = 400
    w_min :: Float64   = 0.0
    w_max :: Float64   = 200.0
end

Base.Broadcast.broadcastable(p::KSParams) = Ref(p)

const ks_params = KSParams()


# Effective labor (stationary distribution over income) #
#-------------------------------------------------------#

"""
Stationary-distribution-weighted aggregate labor for the income chain
`P_y`: solve `π' P_y = π'` (with `Σπ = 1`) and return `Σ π_i y_i`. Used
by `ks_prices` so the Cobb-Douglas wage and rental rate reflect the
true effective labor supply when the income process has a non-trivial
employed/unemployed split.
"""
function ks_effective_labor(P_y::AbstractMatrix, y_grid::AbstractVector)
    n = size(P_y, 1)
    A = P_y' - I(n)
    A[end, :] .= 1.0
    rhs = zeros(n); rhs[end] = 1.0
    π = A \ rhs
    return sum(y_grid .* π)
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached K-S household chain
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`.
"""
function ks_household(p = ks_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = IncomeStage(layout) # defaults: (; axis = :wealth)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.γ)),
    ) # defaults: (; axis = :wealth, utility_axes = nothing, skip_monotonicity_check = false)

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Production prices (plain function, no AbstractBlock) #
#------------------------------------------------------#

"Cobb-Douglas factor prices at aggregate capital `K` and TFP `A`."
function ks_prices(K::Real, A::Real, p = ks_params)
    (; α, δ) = p
    L = ks_effective_labor(p.P_y, p.y_grid)
    r = α * A * (K / L)^(α - 1) - δ
    w = (1 - α) * A * (K / L)^α
    return (; r, w)
end
