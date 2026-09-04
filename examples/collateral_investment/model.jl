####################################################################
# Collateral-constrained investment — Moll (2014) / Buera–Shin (2013) #
####################################################################

# Entrepreneurs operate capital subject to a COLLATERAL CONSTRAINT: a firm with
# net worth `a` can run capital only up to a multiple `λ` of its net worth,
# `k ≤ λ·a`. Financial frictions therefore cause capital MISALLOCATION — a
# productive but poor entrepreneur cannot reach its efficient scale — and the
# only escape is SELF-FINANCING (accumulating net worth out of retained profit).
#
# The clean computational form (Moll 2014): capital is RENTED within the period,
# so capital is NOT a state variable — it is a STATIC within-period choice
# bounded by collateral, with a CLOSED FORM. The entrepreneur with net worth `a`
# and productivity `z` solves the static profit max
#
#     π(a, z) = max_{k ≤ λ·a} [ z·k^α − (r+δ)·k ],
#
# whose unconstrained optimum is `k* = (α z / (r+δ))^{1/(1−α)}`; the firm uses
# `k = min(k*, λ·a)`. When `λ·a < k*` the COLLATERAL CONSTRAINT BINDS (`k = λa`) —
# this is the friction. The only carried state is net worth `a` (plus the
# productivity `z`), and net worth accumulates by ordinary saving out of
# cash-on-hand `(1+r)·a + π(a, z)`.
#
# So the firm block is the AIYAGARI SAVINGS SPINE with the profit π replacing
# labor income `w·y`, and the collateral cap folded INTO the profit closure —
# a composition of EXISTING library stages, NO bespoke stage:
#
#     Produce ∘ SaveNetWorth ∘ ProductivityShock
#   = WealthChangeStage(cash = (1+r)a + π(a,z)) ∘ ConsumptionSavingsStage(:wealth) ∘ MarkovStage(:z)
#
# `Produce`       — `WealthChangeStage` on `:wealth` (= net worth `a`): maps net worth to
#                   cash-on-hand `(1+r)·a + π(a, z)`. Its `wealth_post` closure reads BOTH
#                   the net-worth axis and the productivity axis `:z` and `env`, and computes
#                   `π(a, z)` WITH the `min(k*, λa)` collateral cap inline. This is the
#                   load-bearing stage: the entire collateral mechanism lives in this closure.
# `SaveNetWorth`  — `ConsumptionSavingsStage` on `:wealth`: from cash-on-hand the firm chooses
#                   next net worth `a'`, consuming `c = cash − a'` with CRRA felicity. This is
#                   the self-financing margin — the only way to relax a binding constraint.
# `ProductivityShock` — `MarkovStage` on `:z`: the AR(1)-in-logs productivity transitions
#                   (Rouwenhorst-discretized), drawn at period END for next period.
#
# Why `MarkovStage` is LAST (cf. aiyagari, where it is first). With the shock last in
# TIME order, productivity `z` is realized at the start of each period (carried in from the
# previous period's end-draw), production uses contemporaneous `(a, z)`, and the
# end-of-chain stationary Λ is — by stationarity — exactly the joint of (net worth,
# current productivity) AT THE PRODUCTION POINT. So every production moment (capital
# operated, fraction constrained, MPK dispersion) reads directly off `at_end`, with no
# need to push Λ through the Markov kernel by hand. Backward induction is unaffected: the
# saver's continuation is `E[V(a',z') | z]` either way.
#
# The net-worth grid is log-spaced: dense near the bottom (where the collateral
# constraint binds hardest and policies are most nonlinear), coarse at the top. The grid
# top must sit far enough out that post-receipt cash `(1+r)a + π` stays in-grid for all
# active cells (else `WealthChangeStage`'s backward interpolation clamps and distorts V).
#
# Prices (`r`) and the cross-sectional WEALTH DISTRIBUTION as an aggregate state are the
# CALLER's outer loop. Here `r` is exogenous (partial equilibrium): no market to clear, so
# the outer loop is a single `solve_steady_state_given_env!`. In general equilibrium one
# would clear the capital/bond market for `r` (and a wage, with a labor block) as in
# aiyagari's tatonnement, and the wealth distribution would be the aggregate state.

using HouseholdStages
using LinearAlgebra: dot


# Parameters #
#------------#

@kwdef struct CollateralInvestmentParams
    α     :: Float64 = 0.33                 # capital share / curvature of z·k^α
    δ     :: Float64 = 0.06                 # capital depreciation
    r     :: Float64 = 0.04                 # exogenous rental/saving rate (PE) ⇒ user-cost r+δ
    λ     :: Float64 = 1.5                  # collateral multiple: k ≤ λ·a (λ→∞ frictionless, λ small severe)
    β     :: Float64 = 0.93                 # discount factor; β(1+r) < 1 ⇒ stationary net-worth distribution
    σ     :: Float64 = 1.5                  # CRRA curvature of consumption felicity
    # Idiosyncratic productivity z: AR(1) in logs, log z' = ρ log z + σ ε, normalized to mean 1.
    ρ_z   :: Float64 = 0.90                 # persistence of log productivity
    σ_z   :: Float64 = 0.30                 # innovation std of log productivity
    N_z   :: Int     = 7                    # number of productivity states
    N_a   :: Int     = 600                  # net-worth grid points
    a_min :: Float64 = 0.05                 # net-worth floor (a > 0 so every firm operates some capital)
    a_max :: Float64 = 300.0                # net-worth ceiling (far out so cash stays in-grid for active cells)
end

Base.Broadcast.broadcastable(p::CollateralInvestmentParams) = Ref(p)

const collateral_investment_params = CollateralInvestmentParams()


# Productivity process — Rouwenhorst discretization of the AR(1) in logs #
#------------------------------------------------------------------------#

"""
Rouwenhorst discretization of an AR(1) `x' = ρ x + σ ε` into `n` states.
Returns `(grid, P)` with `grid` the evenly spaced state values (symmetric about 0,
half-width `σ·√(n−1)/√(1−ρ²)`) and `P` the `n×n` ROW-stochastic transition matrix.
Rouwenhorst is accurate for highly persistent processes (unlike Tauchen at large ρ).
"""
function rouwenhorst(ρ::Real, σ::Real, n::Integer)
    n == 1 && return ([0.0], reshape([1.0], 1, 1))
    p = (1 + ρ) / 2
    P = [p (1 - p); (1 - p) p]
    for m in 3:n
        Pprev = P
        P = zeros(m, m)
        P[1:m-1, 1:m-1] .+= p .* Pprev
        P[1:m-1, 2:m]   .+= (1 - p) .* Pprev
        P[2:m, 1:m-1]   .+= (1 - p) .* Pprev
        P[2:m, 2:m]     .+= p .* Pprev
        P[2:m-1, :]     ./= 2                 # interior rows double-counted
    end
    ψ    = σ * sqrt((n - 1) / (1 - ρ^2))      # half-width of the state space
    grid = collect(range(-ψ, ψ; length = n))
    return (grid, P)
end

"""
Stationary distribution of a row-stochastic matrix `P` by power iteration on `πᵀP = πᵀ`.
Used to normalize the productivity grid to mean 1 under its own ergodic measure.
"""
function stationary_distribution(P::AbstractMatrix; tol = 1e-14, max_iter = 100_000)
    n = size(P, 1)
    π = fill(1 / n, n)
    for _ in 1:max_iter
        πn = P' * π
        maximum(abs, πn .- π) < tol && return πn ./ sum(πn)
        π = πn
    end
    return π ./ sum(π)
end


# Collateral-constrained static profit problem (CLOSED FORM) #
#------------------------------------------------------------#

"""
Unconstrained optimal rented capital `k* = (α z / (r+δ))^{1/(1−α)}` — the scale a firm of
productivity `z` would operate absent any collateral limit. `env` carries `α, r, δ`.
"""
k_unconstrained(z, env) = (env.α * z / (env.r + env.δ))^(1 / (1 - env.α))

"""
Capital actually operated under the collateral constraint, `k = min(k*, λ·a)`: the firm
runs its unconstrained optimum when it can post enough collateral, otherwise it is capped
at `λ·a` (the constraint BINDS). `a` is net worth, `env` carries `α, r, δ, λ`.
"""
k_operated(a, z, env) = min(k_unconstrained(z, env), env.λ * a)

"""
Static operating profit `π(a, z) = z·k^α − (r+δ)·k` at the collateral-feasible capital
`k = min(k*, λa)` — flow profit net of the rental user cost. Added to net worth (with the
safe return) to form cash-on-hand in the `WealthChangeStage`.
"""
function operating_profit(a, z, env)
    k = k_operated(a, z, env)
    return z * k^env.α - (env.r + env.δ) * k
end

"""
Marginal product of capital `αz·k^{α−1}` at the operated capital `k = min(k*, λa)`. In the
frictionless allocation every firm sets `k = k*` and MPK equals the common user cost `r+δ`;
under binding constraints MPK exceeds `r+δ` and DISPERSES across firms — the cross-sectional
spread of (log) MPK is the standard measure of capital misallocation.
"""
mpk(a, z, env) = env.α * z * k_operated(a, z, env)^(env.α - 1)


# Firm chain assembly — THREE library stages, NO bespoke stage #
#--------------------------------------------------------------#

"""
The `(net worth, productivity)` layout: a log-spaced net-worth axis `:wealth` and a discrete
productivity axis `:z` carrying `exp(log_z)` normalized to ergodic mean 1. Returns
`(layout, P_z)` so the chain builder and drivers share one grid construction.
"""
function collateral_investment_layout(p = collateral_investment_params)
    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_raw      = exp.(log_z)
    z_grid     = z_raw ./ dot(stationary_distribution(P_z), z_raw)   # normalize ergodic mean to 1
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log),
        :z      => Discrete(z_grid),
    )
    return (layout, P_z)
end

"""
Build the collateral-constrained investment firm block
`WealthChangeStage(produce) ∘ ConsumptionSavingsStage(:wealth) ∘ MarkovStage(:z)`,
with moments for mean net worth, mean capital operated, mean profit, mean productivity,
the FRACTION OF FIRMS WHOSE COLLATERAL CONSTRAINT BINDS, and the first two moments of
log-MPK (for the misallocation dispersion) attached.

The collateral mechanism lives entirely in the `WealthChangeStage`'s `wealth_post` closure
`(a, z) ↦ (1+r)·a + π(a, z)`, where `π` caps capital at `min(k*, λa)`. The productivity grid
is `exp(log_z)` normalized to ergodic mean 1.
"""
function collateral_investment_household(p = collateral_investment_params)
    layout, P_z = collateral_investment_layout(p)

    produce = WealthChangeStage(layout;          # net worth ↦ cash-on-hand, collateral cap folded in
        axis        = :wealth,
        wealth_post = (; wealth, z, env) -> (1 + env.r) * wealth + operating_profit(wealth, z, env))
    savings = ConsumptionSavingsStage(layout;    # cash-on-hand ↦ next net worth a'
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth)
    shock   = MarkovStage(layout; axis = :z, transition_matrix = P_z)   # z drawn at period end

    firm = produce ∘ savings ∘ shock
    return define_moments!(firm;
        mean_a           = at_end(integrand = :wealth, reduce = sum),
        mean_k           = at_end(integrand = (; wealth, z, env) -> k_operated(wealth, z, env), reduce = sum),
        mean_profit      = at_end(integrand = (; wealth, z, env) -> operating_profit(wealth, z, env), reduce = sum),
        mean_z           = at_end(integrand = :z, reduce = sum),
        frac_constrained = at_end(integrand = (; wealth, z, env) -> env.λ * wealth < k_unconstrained(z, env) ? 1.0 : 0.0, reduce = sum),
        mean_log_mpk     = at_end(integrand = (; wealth, z, env) -> log(mpk(wealth, z, env)), reduce = sum),
        mean_log_mpk_sq  = at_end(integrand = (; wealth, z, env) -> log(mpk(wealth, z, env))^2, reduce = sum))
end


# Exogenous environment (plain function, partial equilibrium) #
#-------------------------------------------------------------#

"Exogenous env consumed by the profit closure and moments: rate, depreciation, collateral, share."
collateral_investment_env(p = collateral_investment_params) = (; r = p.r, δ = p.δ, λ = p.λ, α = p.α)
