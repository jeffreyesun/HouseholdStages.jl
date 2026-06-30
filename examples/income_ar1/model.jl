###########################################################
# AR(1) income via Rouwenhorst — offline discretization    #
###########################################################
#
# The standard incomplete-markets self-insurance model, but with the
# idiosyncratic income process specified as a continuous log-AR(1):
#
#     log y_t = ρ · log y_{t-1} + ε_t,   ε_t ~ N(0, σ_ε²).
#
# The POINT of this example is the split of labour between an OFFLINE
# pre-computation and an IN-PACKAGE stage:
#
#   * The discretization of the AR(1) into an N-state Markov chain
#     (Rouwenhorst 1995 / Tauchen 1986) is a plain numerical routine —
#     `rouwenhorst` / `tauchen` below. It runs ONCE, outside the package,
#     and returns a grid `y_grid` and a constant row-stochastic matrix `T`.
#     It is NOT a stage; it never touches a layout, a kernel, or a value
#     function.
#   * The package only ever sees the FINISHED constant matrix, fed to a
#     plain `MarkovStage(:income; transition_matrix = T)`. From the
#     package's point of view this is byte-identical to Aiyagari's
#     hand-written 3×3 `P_y` — the chain is the canonical spine
#
#         MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage.
#
# So "AR(1) income" is not a new capability of the household block; it is
# a calibration choice resolved entirely in user space before the block is
# built. The package's `MarkovStage` is agnostic about where its matrix
# came from.

using HouseholdStages
using LinearAlgebra: dot


# Offline AR(1) discretization (NOT a stage) #
#--------------------------------------------#

"""
Rouwenhorst (1995) discretization of the log-AR(1) `log y' = ρ log y + ε`,
`ε ~ N(0, σ_ε²)`, into an `N`-state Markov chain. Returns `(y_grid, T)` with
`y_grid` the levels (exp of the log nodes, mean-normalized to 1) and `T` the
`N×N` row-stochastic transition `T[i,j] = P(i→j)`.

Rouwenhorst matches the conditional and unconditional first two moments of
the AR(1) exactly and, unlike Tauchen, stays accurate as `ρ → 1` — the
reason it is the default for persistent income calibrations. The transition
is built by the standard recursion on the `(1+ρ)/2` mixing probability; the
log nodes are equally spaced on `±σ_ε √((N-1)/(1-ρ²))`, the unconditional
std of the chain.
"""
function rouwenhorst(ρ::Real, σ_ε::Real, N::Int)
    p = (1 + ρ) / 2
    # Recursive construction of the transition matrix.
    Θ = [p (1-p); (1-p) p]
    for n in 3:N
        Θ_prev = Θ
        z = zeros(n, n)
        z[1:n-1, 1:n-1] .+= p .* Θ_prev
        z[1:n-1, 2:n]   .+= (1-p) .* Θ_prev
        z[2:n, 1:n-1]   .+= (1-p) .* Θ_prev
        z[2:n, 2:n]     .+= p .* Θ_prev
        z[2:n-1, :]    ./= 2          # interior rows double-counted; renormalize
        Θ = z
    end
    # Log nodes equally spaced on ±ψ, ψ the unconditional std of the chain.
    ψ        = σ_ε * sqrt((N - 1) / (1 - ρ^2))
    log_grid = range(-ψ, ψ; length = N)
    y_grid   = exp.(collect(log_grid))
    # Normalize levels to unconditional mean 1 under the stationary dist.
    π        = stationary_dist(Θ)
    y_grid ./= dot(π, y_grid)
    return y_grid, Θ
end

"""
Tauchen (1986) discretization of the same log-AR(1), provided as an
alternative to `rouwenhorst`. Places `N` equally-spaced log nodes on
`±m·σ_y` (`σ_y` the unconditional std, `m = 3`) and integrates the normal
innovation density over the midpoint bins to fill each row. Less accurate
than Rouwenhorst at high `ρ`; included to show the discretization choice is
a free offline swap that the package never sees.
"""
function tauchen(ρ::Real, σ_ε::Real, N::Int; m::Real = 3.0)
    σ_y  = σ_ε / sqrt(1 - ρ^2)
    z    = range(-m * σ_y, m * σ_y; length = N)
    step = z[2] - z[1]
    Φ(x) = 0.5 * (1 + erf_approx(x / sqrt(2)))
    T = zeros(N, N)
    for i in 1:N
        μ = ρ * z[i]
        for j in 1:N
            if j == 1
                T[i, j] = Φ((z[1] - μ + step/2) / σ_ε)
            elseif j == N
                T[i, j] = 1 - Φ((z[N] - μ - step/2) / σ_ε)
            else
                T[i, j] = Φ((z[j] - μ + step/2) / σ_ε) - Φ((z[j] - μ - step/2) / σ_ε)
            end
        end
        T[i, :] ./= sum(T[i, :])
    end
    y_grid = exp.(collect(z))
    π      = stationary_dist(T)
    y_grid ./= dot(π, y_grid)
    return y_grid, T
end

"""
Abramowitz-Stegun rational approximation of `erf`, so `tauchen` needs no
SpecialFunctions dependency. Max abs error ≈ 1.5e-7 — far finer than the
discretization error it feeds.
"""
function erf_approx(x::Real)
    t = 1 / (1 + 0.3275911 * abs(x))
    y = 1 - (((((1.061405429t - 1.453152027)t) + 1.421413741)t - 0.284496736)t + 0.254829592) * t * exp(-x^2)
    return sign(x) * y
end

"""
Stationary distribution of a row-stochastic `T` by power iteration from
uniform — used only offline to mean-normalize the discretized income grid.
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


# Parameters #
#------------#

@kwdef struct IncomeAR1Params
    β   :: Float64 = 0.96
    σ   :: Float64 = 2.0
    r   :: Float64 = 0.03            # fixed exogenous return, < 1/β − 1 ≈ 0.0417
    w   :: Float64 = 1.0             # fixed wage (level of the endowment)
    # AR(1) income process.
    ρ   :: Float64 = 0.95            # persistence
    σ_ε :: Float64 = 0.20            # innovation std
    N_y :: Int     = 7               # number of discretized income states
    # Wealth grid.
    N_w   :: Int     = 250
    w_min :: Float64 = 0.0
    w_max :: Float64 = 80.0
end

Base.Broadcast.broadcastable(p::IncomeAR1Params) = Ref(p)

const income_ar1_params = IncomeAR1Params()


# Household chain assembly #
#--------------------------#

"""
Build the AR(1)-income household block. The income process is discretized
OFFLINE by `rouwenhorst` into `(y_grid, T)`; the package then sees the plain
constant-matrix spine

    MarkovStage(:income; transition_matrix = T) ∘ IncomeStage ∘ ConsumptionSavingsStage,

byte-identical to Aiyagari's hand-written chain. `A_mean = ∫ wealth dΛ`
(aggregate self-insurance buffer) is the attached moment.
"""
function income_ar1_household(p = income_ar1_params; method = :rouwenhorst)
    y_grid, T = method === :tauchen ? tauchen(p.ρ, p.σ_ε, p.N_y) :
                                      rouwenhorst(p.ρ, p.σ_ε, p.N_y)

    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = T)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        A_mean = at_end(integrand = :wealth, reduce = sum),
    )
end

"Env for the fixed-r partial-equilibrium experiment: the return `r` and wage `w`."
income_ar1_env(p = income_ar1_params) = (; r = p.r, w = p.w)
