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
# The within-period problem decomposes into six stages, in time order (left → right):
#
#     QuitChoice ∘ FlowUtility ∘ Discount ∘ ZShock ∘ Separation ∘ Matching
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
# - `Separation ∘ Matching` — ONE library call: `SearchMatchingStage` (derived sugar)
#   expands to `MarkovStage(separation) ∘ MixingStage(job-search lottery)`; chains
#   flatten, so the two leaves are unchanged. `Separation`: the employed separate
#   EXOGENOUSLY at rate `δ` (transition `[1 0; δ 1−δ]`), BEFORE matching — a worker
#   separated this period searches this same period, so job loss and the search
#   response are not staggered across periods. `Matching`: the
#   unemployed CHOOSE their job-finding probability `p ∈ [0, 1]` directly — the
#   lottery over "search succeeds" (`[0 1; 0 1]`) and "search fails" (identity); the
#   employed rows coincide, so the employed choice is degenerate (`p* = 0`, cost 0).
#   The sugar single-homes the convex UTILS cost `c(p) = κ_s·((1−p)log(1−p) + p)` and
#   its closed-form argmax `p*(y) = 1 − exp(−y/κ_s)`; the scale `κ_s = χ/(A_match·θ)`
#   is calibrated so `c′(p) = χ·e(p)` at the effort `e(p) = −log(1−p)/(A·θ)` the
#   matching technology `p = 1 − exp(−A·e·θ)` requires — higher tightness ⇒ cheaper
#   search. (Partial equilibrium: `θ` is a fixed scalar passed as `tightness`.)
#
# This example PAIRS with `examples/mccall_search`: both use the same gated
# `ArgmaxStage(:emp)` `-Inf` trick with a spectator axis driving a threshold, in
# OPPOSITE directions — there the unemployed choose to ACCEPT (→ employed); here the
# employed choose to QUIT (→ unemployed). References: Mortensen & Pissarides (1994);
# Pissarides (2000) "Equilibrium Unemployment Theory".
#
# Library stages used (NO bespoke household stage): ArgmaxStage, UtilityStage,
# TimeDiscountingStage, MarkovStage, SearchMatchingStage (derived sugar =
# MarkovStage ∘ MixingStage).

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

    # Search cost + matching: κ_s = χ/(A_match·θ) scales the probability-space
    # search cost c(p) = κ_s·((1−p)log(1−p) + p).
    χ       :: Float64 = 0.5       # search-cost scale (effort disutility 0.5·χ·e²)
    A_match :: Float64 = 0.5       # matching efficiency in p(e, θ) = 1 − exp(−A·e·θ)
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
`QuitChoice ∘ FlowUtility ∘ Discount ∘ ZShock ∘ Separation ∘ Matching`, with the
employment rate, the employed/unconditional `z` mass, and the (employed-origin) `z`
mass attached as moments. The `:emp` axis is 1 = unemployed, 2 = employed; `w`, `b_u`
ride `env`; the tightness `θ` and exogenous separation `δ` are fixed params. Six
existing library leaves, no bespoke household stage: the keep/quit choice is a gated
`ArgmaxStage` whose `-Inf` entry forbids self-promotion (the `:z` spectator delivers
the reservation-productivity cutoff), and separation + job search is the
`SearchMatchingStage` sugar — the separation `MarkovStage` composed with the
`MixingStage` lottery over the "success"/"fail" `:emp` kernels at convex
probability-space cost.
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

    quit_choice  = ArgmaxStage(layout; axis = :emp, reward = quit_reward)
    flow_utility = UtilityStage(layout;
        utility = (; emp, z, env) -> emp == :emp ? u_crra(env.w * z, Val(p.σ)) :
                                                   u_crra(env.b_u, Val(p.σ)))
    discount = TimeDiscountingStage(layout; β = p.β)
    z_shock  = MarkovStage(layout; axis = :z, transition_matrix = T_z)

    search_and_match = SearchMatchingStage(layout;  # Separation ∘ Matching (derived sugar):
        separation          = p.δ,                  #   the separated search this same period
        effort_cost_scale   = p.χ,
        matching_efficiency = p.A_match,
        tightness           = p.θ,                  # fixed scalar (partial equilibrium)
    )

    hh = quit_choice ∘ flow_utility ∘ discount ∘ z_shock ∘ search_and_match   # 6 leaves
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
