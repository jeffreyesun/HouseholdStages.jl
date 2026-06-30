######################################################################
# Endogenous separations (Mortensen–Pissarides quit leg) — household #
######################################################################

# A search-and-matching household with ENDOGENOUS separations, built ENTIRELY from
# existing HouseholdStages library stages — no bespoke household stage. The state is
# `(:z, :emp)`: `z` is match productivity (a persistent Markov chain), `:emp` is the
# 2-level labor state. An employed worker whose match productivity has deteriorated
# chooses whether to KEEP the match or QUIT into unemployment; this endogenous-quit
# margin sits ON TOP OF the standard search-matching inflow (effort search → jobs) and
# an exogenous separation rate `δ`. Quits occur at low `z` — the Mortensen–Pissarides
# (1994) reservation-productivity margin.
#
# The within-period problem decomposes into five stages, in time order (left → right):
#
#     QuitChoice ∘ FlowUtility ∘ Discount ∘ ZShock ∘ Matching
#
# - `QuitChoice` (`ArgmaxStage` on the 2-level `:emp` axis) — the (max,+) keep/quit
#   choice. The reward `M[after, before]` GATES with `-Inf`: from an EMPLOYED origin
#   both "stay employed" and "quit → unemployed" are open; from an UNEMPLOYED origin
#   only "stay unemployed" is open (`-Inf` on unemployed → employed — an unemployed
#   worker cannot self-promote; hiring is the matching stage's job). The `:z` axis is
#   a SPECTATOR, so the employed continuation `V_end[emp]` rises with `z` while the
#   unemployed continuation `V_end[unemp]` does not — quit iff `z` is low enough.
# - `FlowUtility` (`UtilityStage`) — flow `u(w·z)` if employed (wage rises in match
#   productivity), `u(b_u)` if unemployed. `w`, `b_u` ride `env`.
# - `Discount` (`TimeDiscountingStage`) — `V_start = β·V_end`.
# - `ZShock` (`MarkovStage` on `:z`) — match productivity follows a persistent
#   (Rouwenhorst-discretized AR(1)) chain. New hires inherit their current `z` cell
#   (stylized — matching does not touch `:z`).
# - `Matching` (`SearchMatchingStage` on `:emp`) — the unemployed choose search effort
#   `e`, pay `cost(e)`, and find a job w.p. `job_finding(e, θ)` at fixed tightness `θ`;
#   the employed separate EXOGENOUSLY at rate `δ`. (Partial equilibrium: `θ` fixed.)
#
# This example PAIRS with `examples/mccall_search`: both use the same gated
# `ArgmaxStage(:emp)` `-Inf` trick with a spectator axis driving a threshold, in
# OPPOSITE directions — there the unemployed choose to ACCEPT (→ employed); here the
# employed choose to QUIT (→ unemployed). References: Mortensen & Pissarides (1994);
# Pissarides (2000) "Equilibrium Unemployment Theory".
#
# Library stages used (NO bespoke household stage): ArgmaxStage, UtilityStage,
# TimeDiscountingStage, MarkovStage, SearchMatchingStage.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct EndogenousSeparationsParams
    β   :: Float64 = 0.96          # discount factor
    σ   :: Float64 = 2.0           # CRRA curvature
    w   :: Float64 = 1.0           # wage scale (employed earn w·z)
    b_u :: Float64 = 0.75          # flow income while unemployed
    δ   :: Float64 = 0.05          # EXOGENOUS separation rate (on top of endogenous quits)

    # Match-productivity chain z (Rouwenhorst-discretized AR(1) in logs).
    N_z   :: Int     = 5
    ρ_z   :: Float64 = 0.90        # persistence
    σ_z   :: Float64 = 0.20        # innovation std (logs)

    # Search effort + matching.
    N_eff   :: Int     = 10
    eff_max :: Float64 = 3.0
    χ       :: Float64 = 0.5       # effort-cost scale: cost(e) = 0.5·χ·e²
    A_match :: Float64 = 0.5       # matching efficiency in p(e,θ) = 1 − exp(−A·e·θ)
    θ       :: Float64 = 2.5       # FIXED market tightness (partial equilibrium)
end

Base.Broadcast.broadcastable(p::EndogenousSeparationsParams) = Ref(p)

const endogenous_separations_params = EndogenousSeparationsParams()


# Match-productivity chain (plain economic data) #
#------------------------------------------------#

"""
Rouwenhorst (1995) discretization of a log-AR(1) `x' = ρ·x + ε`, `ε ~ N(0, σ²)`, on
`n` nodes. Returns `(T, log_grid)`: the row-stochastic transition `T[from, to]` and the
evenly spaced log-grid `[-ψ, ψ]` with `ψ = σ/√(1−ρ²)·√(n−1)`. The Rouwenhorst method
matches the AR(1) persistence exactly and is the standard choice for highly persistent
chains.
"""
function rouwenhorst(n::Int, ρ::Real, σ::Real)
    p = (1 + ρ) / 2
    Θ = [p (1 - p); (1 - p) p]
    for i in 3:n
        Θp = Θ
        Θ  = zeros(i, i)
        @views begin
            Θ[1:i-1, 1:i-1] .+= p       .* Θp
            Θ[1:i-1, 2:i]   .+= (1 - p) .* Θp
            Θ[2:i,   1:i-1] .+= (1 - p) .* Θp
            Θ[2:i,   2:i]   .+= p       .* Θp
        end
        Θ[2:i-1, :] ./= 2              # interior rows double-count; renormalize
    end
    σ_x  = σ / sqrt(1 - ρ^2)
    ψ    = σ_x * sqrt(n - 1)
    grid = collect(range(-ψ, ψ; length = n))
    return Θ, grid
end

"""
The match-productivity transition `T_z` and level grid `z` for the model: Rouwenhorst
on `(N_z, ρ_z, σ_z)`, exponentiated to levels and rescaled so the chain's stationary
mean productivity is 1 (the Rouwenhorst stationary law is `Binomial(N_z−1, 1/2)`).
"""
function z_process(p = endogenous_separations_params)
    T, lg = rouwenhorst(p.N_z, p.ρ_z, p.σ_z)
    z      = exp.(lg)
    n      = p.N_z
    π      = [binomial(n - 1, k) for k in 0:n-1] ./ 2.0^(n - 1)   # stationary law
    z    ./= sum(π .* z)                                          # normalize E[z] = 1
    return T, z
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached endogenous-separations block
`QuitChoice ∘ FlowUtility ∘ Discount ∘ ZShock ∘ Matching`, with the employment rate,
the employed/unconditional `z` mass, and the (employed-origin) `z` mass attached as
moments. The `:emp` axis is 1 = unemployed, 2 = employed; `w`, `b_u` ride `env`; the
tightness `θ` and exogenous separation `δ` are fixed params. Five existing library
stages, no bespoke household stage: the keep/quit choice is a gated `ArgmaxStage`
whose `-Inf` entry forbids self-promotion, and the `:z` spectator delivers the
reservation-productivity (quit) cutoff.
"""
function endogenous_separations_household(p = endogenous_separations_params)
    T_z, z = z_process(p)
    layout = GriddedLayout(
        :z   => Discrete(z),
        :emp => Discrete([:unemp, :emp]),   # 1 = unemployed, 2 = employed
    )

    # Keep/quit reward M[after, before] on :emp (rows = destination, cols = origin).
    #   origin unemp (col 1): only stay unemp (M[1,1]=0); cannot self-promote (M[2,1]=-Inf)
    #   origin emp   (col 2): stay employed (M[2,2]=0) OR quit→unemp (M[1,2]=0)   — both open
    quit_reward = [0.0   0.0;
                   -Inf  0.0]

    efforts = collect(range(0.0, p.eff_max; length = p.N_eff))

    quit_choice  = ArgmaxStage(layout; axis = :emp, reward = quit_reward, search = :brute)
    flow_utility = UtilityStage(layout;
        utility = (; emp, z, env) -> emp == :emp ? u_crra(env.w * z, Val(p.σ)) :
                                                   u_crra(env.b_u, Val(p.σ)))
    discount = TimeDiscountingStage(layout; β = p.β)
    z_shock  = MarkovStage(layout; axis = :z, transition_matrix = T_z)
    matching = SearchMatchingStage(layout; axis = :emp,
        efforts     = efforts,
        cost        = e -> 0.5 * p.χ * e^2,
        job_finding = (e, θ) -> 1 - exp(-p.A_match * e * θ),
        separation  = p.δ,
        tightness   = p.θ)            # fixed scalar tightness (partial equilibrium)

    hh = quit_choice ∘ flow_utility ∘ discount ∘ z_shock ∘ matching
    return define_moments!(hh;
        employment = at_end(integrand = (; emp) -> emp == :emp ? 1.0 : 0.0, reduce = sum),
        z_emp      = at_end(integrand = (; z, emp) -> emp == :emp ? z : 0.0, reduce = sum),
        z_uncond   = at_end(integrand = (; z) -> z, reduce = sum),
    )
end

"""
Env for the endogenous-separations partial-equilibrium experiment: the wage scale `w`
and unemployment benefit `b_u` consumed by the flow-utility closure.
"""
endogenous_separations_env(p = endogenous_separations_params) = (; w = p.w, b_u = p.b_u)
