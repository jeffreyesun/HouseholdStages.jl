################################################################
# Diagnostic expectations (Bordalo-Gennaioli-Shleifer 2018)     #
################################################################
#
# Households form DIAGNOSTIC (representativeness-distorted) expectations of
# their income: they over-weight states that have become more likely
# relative to a reference, exaggerating recent news (Bordalo, Gennaioli,
# Shleifer 2018). The distortion is applied to the income transition matrix:
# from a true row-stochastic `T`, with stationary distribution `π`, the
# diagnostic transition tilts each row by the likelihood ratio to a power θ,
#
#     T_diag[y, y'] ∝ T[y, y'] · ( T[y, y'] / π[y'] )^θ,
#
# renormalized per row. θ > 0 is the diagnosticity; θ = 0 recovers rational
# expectations. States that are over-represented under `T[y, ·]` relative to
# the unconditional `π` (the "representative" states) get extra weight.
#
# The distortion is a PLAIN OFFLINE matrix computation — `diagnostic_tilt`
# below. The household solves and simulates under the resulting CONSTANT
# matrix via an ordinary `MarkovStage`; the package never sees the tilt, only
# the finished distorted matrix. The chain is the canonical spine:
#
#   MarkovStage(:income; transition_matrix = T_diag) ∘ IncomeStage ∘ ConsumptionSavingsStage.
#
# (We use the distorted transition for BOTH the belief and the actual law of
# motion — the "fully diagnostic" steady state, the standard BGS exercise.)

using HouseholdStages
using LinearAlgebra: dot


# Offline diagnostic distortion (NOT a stage) #
#---------------------------------------------#

"""
Stationary distribution of a row-stochastic `T` by power iteration from
uniform — used offline to anchor the diagnostic likelihood ratio.
"""
function stationary_dist(T::AbstractMatrix; iters::Int = 10_000, tol = 1e-14)
    n = size(T, 1)
    π = fill(1 / n, n)
    for _ in 1:iters
        π_next = vec(π' * T)
        maximum(abs, π_next - π) < tol && (π = π_next; break)
        π = π_next
    end
    return π ./ sum(π)
end

"""
Diagnostic-expectations tilt (Bordalo-Gennaioli-Shleifer 2018) of a true
row-stochastic transition `T`: returns the row-stochastic

    T_diag[i, j] ∝ T[i, j] · ( T[i, j] / π[j] )^θ,

`π` the stationary distribution of `T`, `θ ≥ 0` the diagnosticity. `θ = 0`
returns `T` unchanged (rational expectations). The representativeness factor
`(T[i,j]/π[j])^θ` over-weights states that are more likely under the
conditional than under the unconditional — the BGS "kernel of truth"
overreaction. Pure matrix algebra; the package only ever sees the result.
"""
function diagnostic_tilt(T::AbstractMatrix, θ::Real)
    θ == 0 && return copy(T)
    π = stationary_dist(T)
    n = size(T, 1)
    Td = similar(T)
    for i in 1:n, j in 1:n
        Td[i, j] = T[i, j] * (T[i, j] / π[j])^θ
    end
    Td ./= sum(Td; dims = 2)
    return Td
end


# Parameters #
#------------#

@kwdef struct DiagnosticParams
    β   :: Float64 = 0.96
    σ   :: Float64 = 2.0
    r   :: Float64 = 0.03                  # fixed, < 1/β − 1 ≈ 0.0417
    w   :: Float64 = 1.0
    θ   :: Float64 = 1.0                   # diagnosticity (0 ⇒ rational expectations)

    # True income process (persistent 5-state, mean ≈ 1).
    y_grid :: Vector{Float64} = [0.5, 0.75, 1.0, 1.33, 2.0]
    P_true :: Matrix{Float64} = [0.70 0.20 0.07 0.02 0.01;
                                 0.15 0.55 0.20 0.07 0.03;
                                 0.05 0.20 0.50 0.20 0.05;
                                 0.03 0.07 0.20 0.55 0.15;
                                 0.01 0.02 0.07 0.20 0.70]

    N_w   :: Int     = 250
    w_min :: Float64 = 0.0
    w_max :: Float64 = 80.0
end

Base.Broadcast.broadcastable(p::DiagnosticParams) = Ref(p)

const diagnostic_params = DiagnosticParams()


# Household chain assembly #
#--------------------------#

"""
Build the diagnostic-expectations household block. The income transition is
distorted OFFLINE by `diagnostic_tilt(P_true, θ)` into a CONSTANT matrix,
then handed to a plain `MarkovStage`:

    MarkovStage(:income; transition_matrix = T_diag) ∘ IncomeStage ∘ ConsumptionSavingsStage.

`θ = 0` recovers rational expectations (same block, untilted matrix). The
attached moment is `A_mean = ∫ wealth dΛ`.
"""
function diagnostic_household(p = diagnostic_params; θ = p.θ)
    T_diag = diagnostic_tilt(p.P_true, θ)

    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = T_diag)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        A_mean = at_end(integrand = :wealth, reduce = sum),
    )
end

"Env for the fixed-r partial-equilibrium experiment: return `r`, wage `w`."
diagnostic_env(p = diagnostic_params) = (; r = p.r, w = p.w)
