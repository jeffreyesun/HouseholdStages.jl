######################################################################
# R&D / intangible-knowledge investment — convex-cost accumulation    #
######################################################################

# A firm accumulating an intangible KNOWLEDGE stock via R&D, subject to a CONVEX
# R&D cost and a stochastic demand/productivity shifter. Textbook smooth-investment
# example in the §1 stock-investment family. The within-period firm block is
# EXISTING library stages, in time order, with NO bespoke stage —
#
#     DemandShock ∘ Revenue ∘ DoRnD
#   = MarkovStage(:shock) ∘ UtilityStage(shock·knowledge^η) ∘ CapitalInvestmentStage(:knowledge)
#
# `DemandShock` — `MarkovStage` on the demand/productivity axis `:shock`: an
#                 AR(1)-in-logs shifter (Rouwenhorst-discretized) scaling revenue.
# `Revenue`     — `UtilityStage` adding flow revenue `shock·knowledge^η` to the
#                 value. The LOAD-BEARING stage: revenue depends on BOTH the
#                 knowledge stock `knowledge` AND the demand state `shock`, and a
#                 `UtilityStage` closure `(; knowledge, shock) -> shock·knowledge^η`
#                 reads both axes. `CapitalInvestmentStage`'s own `(value; env)` closures
#                 see ONLY the operative axis (`knowledge`) and `env`, so the
#                 `shock` dependence CANNOT live in the R&D stage. Splitting the
#                 shock-dependent revenue into a separate `UtilityStage` is what
#                 gives the knowledge stock a NON-degenerate cross-section: R&D
#                 responds to demand purely through the continuation value, and the
#                 persistent `shock` chain spreads the stationary mass over
#                 `(knowledge, shock)`.
# `DoRnD`       — `CapitalInvestmentStage` on `:knowledge`: from `knowledge` the firm picks
#                 next knowledge `knowledge'`, paying a convex cost
#                 `c_rnd·i^{1/γ}` on GROSS R&D `i = knowledge' − (1−δ_z)knowledge`
#                 (`γ < 1` ⇒ exponent `1/γ > 1`, convex) and discounting by `β`.
#                 Production in this stage is 0 (revenue lives in the `UtilityStage`);
#                 only the R&D cost + knowledge depreciation are here.
#
# Why depreciation is needed for a finite optimum. With concave revenue
# `shock·knowledge^η` (η < 1) and a one-time convex R&D cost, undepreciated
# knowledge would be costless to hold forever and the firm would accumulate
# without bound. Knowledge depreciation `δ_z` makes maintaining the stock costly
# every period (holding `knowledge` needs gross R&D `δ_z·knowledge` at cost
# `c_rnd(δ_z·knowledge)^{1/γ}`), so for `1/γ > η` the per-period payoff is
# eventually decreasing and the optimum is finite. `CapitalInvestmentStage` carries
# depreciation in its gross-investment definition `i = knowledge' − (1−δ_z)knowledge`.
#
# The reward `−c_rnd·(knowledge' − (1−δ_z)knowledge)^{1/γ}` is supermodular in
# `(knowledge', knowledge)`, so `CapitalInvestmentStage`'s `:divide_conquer` monotone solve
# is valid. Prices are exogenous (partial equilibrium): the "outer loop" is a
# single stationary `solve_steady_state_given_env!` over `(knowledge, shock)`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct RnDInvestmentParams
    β     :: Float64 = 0.96                 # firm discount factor
    γ     :: Float64 = 0.50                 # R&D-cost curvature ∈ (0,1) ⇒ cost exponent 1/γ = 2 (convex)
    η     :: Float64 = 0.50                 # revenue curvature in knowledge (<1: diminishing returns)
    δ_z   :: Float64 = 0.10                 # knowledge depreciation (intangible obsolescence)
    c_rnd :: Float64 = 2.0                  # R&D cost scale: cost = c_rnd·i^{1/γ}
    # Demand / productivity shifter: AR(1) in logs, log s' = ρ log s + σ ε, mean 1.
    ρ_s   :: Float64 = 0.80                 # persistence of log demand
    σ_s   :: Float64 = 0.30                 # innovation std of log demand
    N_s   :: Int     = 7                    # number of demand states
    N_k   :: Int     = 120                  # knowledge grid points
    k_min :: Float64 = 0.3
    k_max :: Float64 = 20.0
end

Base.Broadcast.broadcastable(p::RnDInvestmentParams) = Ref(p)

const rnd_investment_params = RnDInvestmentParams()


# Demand process — Rouwenhorst discretization of the AR(1) in logs #
#-----------------------------------------------------------------#

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


# Household / firm chain assembly — THREE library stages, NO bespoke stage #
#-------------------------------------------------------------------------#

"""
Build the R&D / knowledge-investment firm block
`MarkovStage(:shock) ∘ UtilityStage(shock·knowledge^η) ∘ CapitalInvestmentStage(:knowledge)`,
with `mean_knowledge`, `mean_revenue`, and `mean_shock` moments attached.

The demand axis `:shock` carries `exp(grid)` (mean log-shock ≈ 0, mean shock ≈ 1).
Revenue `shock·knowledge^η` lives in a `UtilityStage` reading BOTH axes — the R&D
stage's `(value; env)` closures cannot see `shock`, so this split is what gives the
knowledge stock a non-degenerate cross-section over `(knowledge, shock)`.
"""
function rnd_investment_household(p = rnd_investment_params)
    log_s, P_s = rouwenhorst(p.ρ_s, p.σ_s, p.N_s)
    s_grid     = exp.(log_s)

    layout = GriddedLayout(
        :knowledge => GriddedContinuous(p.k_min, p.k_max, p.N_k),
        :shock     => Discrete(s_grid),
    )

    shock   = MarkovStage(layout; axis = :shock, transition_matrix = P_s)
    revenue = UtilityStage(layout;
        utility = (; knowledge, shock) -> shock * knowledge^p.η)        # reads BOTH knowledge and shock
    do_rnd  = CapitalInvestmentStage(layout;
        axis         = :knowledge,
        β            = p.β,
        depreciation = p.δ_z,
        production   = (k; env) -> 0.0,                                 # revenue lives in the UtilityStage
        effort_cost  = (i; env) -> p.c_rnd * i^(1 / p.γ))               # convex R&D cost on gross R&D i ≥ 0
        # defaults: (; monotone_search = :divide_conquer, assume_monotone = false)

    hh = shock ∘ revenue ∘ do_rnd
    return define_moments!(hh;
        mean_knowledge = at_end(integrand = :knowledge, reduce = sum),
        mean_revenue   = at_end(integrand = (; knowledge, shock) -> shock * knowledge^p.η, reduce = sum),
        mean_shock     = at_end(integrand = :shock, reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"Exogenous env: empty (β is baked into the stage; no market to clear)."
rnd_investment_env(p = rnd_investment_params) = (;)
