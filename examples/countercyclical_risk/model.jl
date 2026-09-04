###############################################################
# Countercyclical income risk (Storesletten-Telmer-Yaron 2004) #
###############################################################
#
# Idiosyncratic income risk is HIGHER in recessions: Storesletten, Telmer
# and Yaron (2004) document that the variance of idiosyncratic earnings
# shocks rises sharply when aggregate output is low. Here the income
# transition matrix is a function of the AGGREGATE STATE `env.z` — the
# recession transition spreads more mass to the tail states than the boom
# transition does.
#
# The package expresses this with an ENV-CLOSURE transition: instead of a
# constant matrix, `MarkovStage` is handed `(; env) -> T(env.z)`. The
# `MarkovStage` kernel re-seats its transition whenever `env` changes (the
# §5.3 static-refill contract: an env-dependent field is refilled on the
# first `backward!` after any env change). The chain is otherwise the
# canonical spine:
#
#   MarkovStage(:income; transition_matrix = (; env) -> T(env.z))
#       ∘ IncomeStage ∘ ConsumptionSavingsStage
#
# In a steady state `z` is FIXED. So we solve TWICE — once at `z = :boom`,
# once at `z = :recession` — on the SAME household object, and compare the
# stationary wealth distributions. The only thing that changes between the
# two solves is the env-closure's output; that the package delivers two
# different stationary distributions demonstrates the kernel genuinely
# re-seats `T` when `env.z` changes.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CountercyclicalParams
    β   :: Float64 = 0.96
    σ   :: Float64 = 2.0
    r   :: Float64 = 0.03                  # fixed, < 1/β − 1 ≈ 0.0417
    w   :: Float64 = 1.0

    # Income state levels (mean ≈ 1), a fixed log-spaced grid shared by both
    # aggregate states. Only the CONDITIONAL VARIANCE of the transition
    # changes with the cycle; the grid and the persistence stay put.
    y_grid :: Vector{Float64} = exp.(collect(range(-0.6, 0.6; length = 7)))

    # Same persistence ρ in boom and recession; the recession raises the
    # innovation std σ_ε (countercyclical idiosyncratic risk, StY 2004)
    # while holding ρ fixed. The transition is Tauchen on the FIXED log grid,
    # so a wider σ_ε spreads each conditional draw without moving the grid.
    ρ            :: Float64 = 0.92
    σ_boom       :: Float64 = 0.13
    σ_recession  :: Float64 = 0.26    # 2× the boom innovation std

    N_w   :: Int     = 250
    w_min :: Float64 = 0.0
    w_max :: Float64 = 80.0
end

Base.Broadcast.broadcastable(p::CountercyclicalParams) = Ref(p)

const countercyclical_params = CountercyclicalParams()


# Aggregate-state-dependent income transition (offline matrix builder) #
#---------------------------------------------------------------------#

"""
Abramowitz-Stegun rational approximation of `erf` (max abs error ≈ 1.5e-7),
so the offline Tauchen builder needs no SpecialFunctions dependency.
"""
function erf_approx(x::Real)
    t = 1 / (1 + 0.3275911 * abs(x))
    y = 1 - (((((1.061405429t - 1.453152027)t) + 1.421413741)t - 0.284496736)t + 0.254829592) * t * exp(-x^2)
    return sign(x) * y
end

"""
Income transition for aggregate state `z` (`:boom` / `:recession`): Tauchen
fill of an AR(1) `log y' = ρ log y + ε`, `ε ~ N(0, σ(z)²)`, on the FIXED log
grid `log.(y_grid)`. Persistence `ρ` is the same across the cycle; only the
innovation std `σ(z)` changes — larger in recessions, so each conditional
draw is more dispersed at unchanged persistence (countercyclical
idiosyncratic risk, Storesletten-Telmer-Yaron 2004). Pure matrix algebra;
the package only ever sees the constructed row-stochastic matrix.
"""
function countercyclical_T(z::Symbol, p = countercyclical_params)
    g    = log.(p.y_grid)
    n    = length(g)
    σ    = z === :recession ? p.σ_recession : p.σ_boom
    step = g[2] - g[1]
    Φ(x) = 0.5 * (1 + erf_approx(x / sqrt(2)))
    T = zeros(n, n)
    for i in 1:n
        μ = p.ρ * g[i]
        for j in 1:n
            if j == 1
                T[i, j] = Φ((g[1] - μ + step/2) / σ)
            elseif j == n
                T[i, j] = 1 - Φ((g[n] - μ - step/2) / σ)
            else
                T[i, j] = Φ((g[j] - μ + step/2) / σ) - Φ((g[j] - μ - step/2) / σ)
            end
        end
        T[i, :] ./= sum(T[i, :])
    end
    return T
end


# Household chain assembly #
#--------------------------#

"""
Build the countercyclical-risk household block. The income `MarkovStage` is
handed an ENV-CLOSURE `(; env) -> countercyclical_T(env.z)`, so its
transition re-seats whenever the aggregate state `env.z` changes:

    MarkovStage(:income; transition_matrix = (; env) -> T(env.z))
        ∘ IncomeStage ∘ ConsumptionSavingsStage.

`A_mean = ∫ wealth dΛ` (the aggregate buffer stock) is the attached moment.
"""
function countercyclical_household(p = countercyclical_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income,
        transition_matrix = (; env) -> countercyclical_T(env.z, p))
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

"Env at a fixed aggregate state `z` (`:boom` / `:recession`): return `r`, wage `w`, state `z`."
countercyclical_env(z::Symbol, p = countercyclical_params) = (; r = p.r, w = p.w, z)
