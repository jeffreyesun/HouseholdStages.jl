###############################################################
# Catherine (2022) — countercyclical risk + life-cycle portfolio #
###############################################################

# Catherine (2022): countercyclical (higher-variance, left-skewed in
# recessions) idiosyncratic income risk makes the human-wealth hedge
# state-dependent. The young, who hold most of their wealth as human capital,
# bear the brunt of this cyclical risk — so their precautionary saving and
# their financial portfolio respond to the aggregate state.
#
# The household block is the **CGM life-cycle portfolio chain** (see
# `examples/cocco_gomes_maenhout`) with ONE change: the income `MarkovStage`
# transition is an ENV-CLOSURE `(; env) -> T(env.z)` whose conditional
# variance rises in recessions (the Storesletten-Telmer-Yaron / countercyclical
# Tauchen of `examples/countercyclical_risk`). The chain is otherwise the same
# four existing stages:
#
#     replicate_age(
#         MarkovStage(:income; transition_matrix = (; env) -> T(env.z))
#             ∘ Receipt ∘ ConsumptionSavings ∘ Portfolio,
#         N; axis = :age)
#
# **No bespoke household stage** — this is the CGM chain with the
# countercyclical_risk env-closure swapped onto its income shock.
#
# In a life-cycle solve the aggregate state `z` is fixed over the cohort's
# life, so we solve TWICE on the same object — once with every age facing the
# boom transition, once with every age facing the recession transition — and
# compare the θ*(age) profiles and mean wealth. The recession's wider income
# innovations raise precautionary saving and reshape the risky-share profile.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CatherineParams
    β :: Float64       = 0.96
    σ :: Float64       = 4.0                        # CRRA; Merton interior ≈ 0.28
    N :: Int           = 25                         # life-cycle periods (ages)

    # Idiosyncratic income state: a FIXED log-spaced grid shared by both
    # aggregate states (mean ≈ 1). Only the conditional VARIANCE of the
    # transition changes with the cycle; the grid and persistence stay put.
    y_grid :: Vector{Float64} = exp.(collect(range(-0.5, 0.5; length = 7)))
    ρ            :: Float64 = 0.90
    σ_boom       :: Float64 = 0.12
    σ_recession  :: Float64 = 0.24                  # 2× the boom innovation std (countercyclical risk)

    # Hump-shaped deterministic age earnings (Gourinchas–Parker).
    peak_age   :: Int     = 15
    retire_age :: Int     = 20
    y_peak     :: Float64 = 1.0
    y_curv     :: Float64 = 0.5
    repl       :: Float64 = 0.4

    # Portfolio block (mean premium ≈ 5%).
    R_f     :: Float64 = 1.02
    R_risky :: Vector{Float64} = [0.86, 1.28]
    p_risky :: Vector{Float64} = [0.5, 0.5]
    shares  :: Vector{Float64} = collect(0.0:0.05:1.0)

    N_w   :: Int       = 100
    w_min :: Float64   = 0.0
    w_max :: Float64   = 60.0
    w0_init :: Float64 = 1.5                         # newborn financial endowment (see CGM README)
end

Base.Broadcast.broadcastable(p::CatherineParams) = Ref(p)

const catherine_params = CatherineParams()


# Earnings profile #
#------------------#

"""
Deterministic age-earnings `y(age)` (Gourinchas–Parker hump), as in the CGM
example: downward quadratic peaking at `p.peak_age`, flat retirement
replacement `p.repl·p.y_peak` after `p.retire_age`.
"""
function age_earnings(age::Integer, p = catherine_params)
    age > p.retire_age && return p.repl * p.y_peak
    span = max(p.peak_age - 1, p.N - p.peak_age)
    drop = p.y_curv * ((age - p.peak_age) / span)^2
    return p.y_peak * (1 - drop)
end


# Aggregate-state-dependent income transition (offline Tauchen, from countercyclical_risk) #
#-----------------------------------------------------------------------------------------#

"""
Abramowitz-Stegun rational approximation of `erf` (max abs error ≈ 1.5e-7),
so the offline Tauchen builder needs no SpecialFunctions dependency. Copied
from `examples/countercyclical_risk`.
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
innovation std `σ(z)` changes — larger in recessions (countercyclical
idiosyncratic risk). Pure matrix algebra; the package only ever sees the
constructed row-stochastic matrix. Copied from `examples/countercyclical_risk`.
"""
function catherine_T(z::Symbol, p = catherine_params)
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

"""
Stationary distribution of the income chain at aggregate state `z` — the
newborn draw over the idiosyncratic income state. Power-iterates `catherine_T(z)`.
"""
function income_stationary(z::Symbol, p = catherine_params)
    T = catherine_T(z, p)
    n = size(T, 1)
    π = fill(1 / n, n)
    for _ in 1:10_000
        π_next = T' * π
        maximum(abs, π_next - π) < 1e-14 && (π = π_next; break)
        π = π_next
    end
    return π ./ sum(π)
end


# Household chain assembly #
#--------------------------#

"""
The Catherine household block: the CGM life-cycle portfolio chain with the
income `MarkovStage` transition handed the countercyclical env-closure
`(; env) -> catherine_T(env.z)`. `replicate_age` stacks `N` age copies along
the `:age` axis. Returns the moment-attached `ProductStage`; the finite-horizon
backward+forward sweep lives in `steady_state.jl`.
"""
function catherine_household(p = catherine_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
        :age => Discrete([1]),
    )

    shock   = MarkovStage(layout; axis = :income,
        transition_matrix = (; env) -> catherine_T(env.z, p))                 # countercyclical env-closure
    receipt = WealthChangeStage(layout;                                       # cash-on-hand x = b + y(age)·ε
        wealth_post = (; wealth, income, env) -> wealth + env.y * income)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )
    portfolio = MeanVarianceStage(layout;
        shares = p.shares, risk_free = p.R_f, risky_returns = p.R_risky, probs = p.p_risky)

    age_chain = shock ∘ receipt ∘ savings ∘ portfolio
    hh = replicate_age(age_chain, p.N; axis = :age)
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end
