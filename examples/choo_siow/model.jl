#####################################################################
# One-sided Partner Choice (Choo-Siow 2006, one side) — logit match #
#####################################################################

# Choo & Siow (2006, JPE) show that, under additive type-I extreme-value
# match utility, the equilibrium matching probabilities of a marriage
# market ARE a logit: the log-odds that an agent of a given side matches a
# partner of type k (versus staying single) equal the *systematic* match
# payoff for that pairing. This example takes that insight literally and
# builds the ONE-SIDED problem — the searching side — as a stationary
# household block, with the OTHER side's availability taken as given
# (exogenous, supplied through `env`).
#
# A searcher carries a single discrete state: their marriage state,
#
#     marital ∈ {single, matched-to-type-1, …, matched-to-type-K}.
#
# The within-period problem decomposes into four stages, in time order:
#
#     Flow ∘ Discount ∘ PartnerChoice ∘ Dissolution
#
# Library stages used (NO bespoke household stage in this file):
#   Flow          — `UtilityStage`: per-period flow payoff by marital state
#                   (the single's outside flow `u_single`; a match-to-k's
#                   companionship flow `α[k]`). State-only V-shifter.
#   Discount      — `TimeDiscountingStage`: `V_start = β · V_end`. Supplies
#                   the contraction that makes the stationary value
#                   recursion well-posed.
#   PartnerChoice — `LogitChoiceStage` on the :marital axis. THIS is the
#                   Choo-Siow logit. Singles (origin = single) draw a Gumbel
#                   and pick a destination: stay single, or match type k with
#                   systematic payoff `Π[k]` entered as a NEGATIVE transition
#                   cost `C[single, matched-k] = −Π[k]`. Already-matched
#                   agents are pinned (cost `+Inf` to every destination but
#                   their own, `0` to stay) — they do not re-choose. The
#                   resulting `π(k | single)` is exactly the Choo-Siow
#                   matching probability: `log(π_k / π_0) = Π[k]` at ε = 1.
#   Dissolution   — `MarkovStage` on the :marital axis: each match dissolves
#                   to `single` at exogenous hazard δ; singles stay single.
#
# The cost matrix `C[origin, dest]` is the only place the model's economics
# live, and it is plain data built from primitives by `partner_cost_matrix`
# (an outer-loop helper, not a stage). The systematic payoff `Π[k]` reflects
# the OTHER side: `Π[k] = a[k] + ω · log(n[k])`, with `a[k]` intrinsic
# attractiveness and `n[k]` the exogenous availability of partner type k
# (read from `env`, so a driver can vary the supply of partner types without
# rebuilding the chain). `ω = 1` recovers the canonical Choo-Siow
# "log availability enters the match log-odds one-for-one" form.
#
# The single↔matched flow (singles match in via the logit; matches dissolve
# back via the hazard) makes the marital chain ergodic, so
# `solve_steady_state_given_env!` returns a non-degenerate stationary
# distribution over marriage states — the cross-section of a stationary
# marriage market.
#
# Scope. This is the ONE-sided block: the partner side is exogenous. The
# TWO-sided Choo-Siow equilibrium (where both sides' availabilities clear a
# matching market and the `n[k]` are endogenous) is a known outer-loop gap —
# it would be a fixed point on the availabilities `n[k]`, layered OUTSIDE
# this household block exactly as tatonnement on K layers outside Aiyagari.
# Nothing in the household block changes; only the driver would close it.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct ChooSiowParams
    β :: Float64 = 0.94                         # patience (per-period discount)
    δ :: Float64 = 0.10                         # exogenous match-dissolution hazard
    ε :: Float64 = 1.0                          # Gumbel scale of the match logit (ε = 1 ⇒ canonical Choo-Siow)
    ω :: Float64 = 1.0                          # weight of log-availability in the match payoff (1 ⇒ Choo-Siow)
    # Partner types (the OTHER side). `a[k]` is intrinsic attractiveness of a
    # match to type k; `α[k]` is the per-period companionship flow once matched.
    partner_types :: Vector{Symbol}  = [:k1, :k2, :k3]
    a :: Vector{Float64} = [0.8, 0.4, 0.0]      # intrinsic match attractiveness by type
    α :: Vector{Float64} = [0.6, 0.5, 0.3]      # per-period flow payoff while matched to type k
    u_single :: Float64 = 0.0                   # per-period flow payoff while single (outside option)
end

Base.Broadcast.broadcastable(p::ChooSiowParams) = Ref(p)

const params = ChooSiowParams()


# Marital-state axis values: single + one matched state per partner type #
#-----------------------------------------------------------------------#

"""
The marital-state axis labels: `:single` followed by one `:matched_k…`
symbol per partner type. Position 1 is single; position k+1 is matched to
partner type k. Plain layout data, not a stage.
"""
marital_states(p = params) =
    vcat(:single, Symbol.("matched_", p.partner_types))


# Match payoffs and the logit cost matrix (plain primitives, not stages) #
#------------------------------------------------------------------------#

"""
Systematic Choo-Siow match payoff `Π[k] = a[k] + ω·log(n[k])` for each
partner type, given the exogenous availability vector `n` (length K). This
is the OTHER side entering the searcher's problem: more-available types are
more attractive to match, one-for-one in logs at ω = 1.
"""
match_payoff(n::AbstractVector, p = params) = p.a .+ p.ω .* log.(n)

"""
The `(K+1)×(K+1)` partner-choice transition cost `C[origin, dest]` for the
`LogitChoiceStage`, built from the systematic match payoffs `Π` (length K).

Row `single` (origin 1): cost `0` to stay single, `−Π[k]` to match type k —
so the destination payoff `−C + V_end` carries the Choo-Siow surplus. Rows
`matched-k` (origins 2…K+1): cost `0` to stay put, `+Inf` to move anywhere
else — `exp(−Inf/ε) = 0` pins a matched agent at their own state (they do
not re-choose). Plain data handed to the stage; not household-stage logic.
"""
function partner_cost_matrix(Π::AbstractVector, p = params)
    K = length(p.partner_types)
    n = K + 1
    C = fill(Inf, n, n)
    C[1, 1] = 0.0                       # single → stay single: outside option, normalised to 0
    for k in 1:K
        C[1, k + 1] = -Π[k]             # single → match type k: pay −Π[k] (negative cost = payoff)
        C[k + 1, k + 1] = 0.0           # matched-k → stay matched-k: free
    end
    return C
end

"""
The `(K+1)×(K+1)` row-stochastic dissolution transition `T[from, to]`:
each matched state dissolves to `single` with probability δ (stays matched
with `1−δ`); `single` stays single with probability 1. Plain data for the
`MarkovStage`; not household-stage logic.
"""
function dissolution_matrix(p = params)
    K = length(p.partner_types)
    n = K + 1
    T = zeros(n, n)
    T[1, 1] = 1.0                       # single stays single
    for k in 1:K
        T[k + 1, 1]     = p.δ           # matched-k → single (dissolution)
        T[k + 1, k + 1] = 1 - p.δ       # matched-k stays matched-k
    end
    return T
end


# Flow-utility table (plain primitive, not a stage) #
#---------------------------------------------------#

"""
The per-marital-state flow-payoff vector handed to `UtilityStage`: the
single's outside flow `u_single` in position 1, the companionship flow
`α[k]` of a match to type k in position k+1.
"""
flow_utility(p = params) = vcat(p.u_single, p.α)


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached one-sided Choo-Siow partner-choice household block
`Flow ∘ Discount ∘ PartnerChoice ∘ Dissolution` over the single marital-state
axis. The partner-choice cost is read from `env` (`FromEnv(:cost_matrix)`),
so the systematic match payoffs — and through them the exogenous partner
availabilities — can be varied by a driver without rebuilding the chain.
Moments attached: the single share and the overall match rate.
"""
function choo_siow_household(p = params)
    states = marital_states(p)
    K = length(p.partner_types)
    layout = GriddedLayout(:marital => Discrete(states))

    flow     = UtilityStage(layout; utility = flow_utility(p))
    discount = TimeDiscountingStage(layout; β = p.β)
    choice   = LogitChoiceStage(layout;
        axis        = :marital,
        cost_matrix = FromEnv(:cost_matrix),     # built from match payoffs by the driver
        ε           = p.ε,
    )
    dissolve = MarkovStage(layout; axis = :marital,
                           transition_matrix = dissolution_matrix(p))

    hh = flow ∘ discount ∘ choice ∘ dissolve

    # `marital` position 1 is `:single`; positions 2…K+1 are matched. The
    # integrands are plain indicator dep closures over the `:marital` axis.
    is_single(marital) = marital == :single
    return define_moments!(hh;
        single_share = at_end(integrand = (; marital) -> is_single(marital) ? 1.0 : 0.0,
                              reduce = sum),
        match_rate   = at_end(integrand = (; marital) -> is_single(marital) ? 0.0 : 1.0,
                              reduce = sum),
    )
end


# Building the env (plain function, no AbstractBlock) #
#-----------------------------------------------------#

"""
Assemble the `env` the chain reads, given the exogenous availability vector
`n` (length K) of partner types: the only env field the household block
consumes is the partner-choice `cost_matrix`, built from the Choo-Siow
match payoffs `Π = a + ω·log(n)`. A driver varies `n` to study how the
supply of partner types reshapes the stationary marriage market.
"""
function choo_siow_env(n::AbstractVector, p = params)
    Π = match_payoff(n, p)
    return (; cost_matrix = partner_cost_matrix(Π, p), n)
end
