################################################################
# Hopenhayn (1992) — firm entry & exit, the exit composite + Entry #
################################################################
#
# Household↔firm dictionary used here (§6 of EXAMPLES.md):
#   wealth/income shock ↔ productivity shock `z`      (MarkovStage)
#   death               ↔ firm exit                   (EndogenousExit)
#   bequest/value-of-death ↔ scrap / liquidation value (the REQUIRED `bequest` field)
#   birth               ↔ firm entry                  (EntryStage, the additive Λ += g source)
#
# A firm is an AGENT under this relabelling — nothing in the V/Λ machinery knows
# household from firm. The within-period firm block is EXISTING library stages, in
# time order, with NO bespoke stage:
#
#     Entry ∘ Profit ∘ Exit ∘ Discount ∘ Shock
#   = EntryStage(g) ∘ UtilityStage(π(z)) ∘ EndogenousExit(scrap) ∘ TimeDiscountingStage(β) ∘ MarkovStage(:z)
#
# The backward (value) sweep runs the chain right-to-left, reproducing the Hopenhayn
# recursion exactly:
#     V(z) = π(z) + max{ scrap , β·E[V(z')|z] }.
#   `Shock`     — MarkovStage backward forms the continuation expectation E[V(z')|z].
#   `Discount`  — TimeDiscountingStage scales it by β = 1/(1+r).
#   `Exit`      — EndogenousExit takes max(continuation, scrap): a firm exits (taking
#                 the scrap value) when its discounted continuation falls below scrap.
#                 This IS optimal stopping — the §5(i) keep-vs-stop ArgmaxStage that the
#                 exit composite wraps over a transient `:exiting` axis.
#   `Profit`    — UtilityStage adds the per-period operating profit π(z) (the static
#                 labour choice is closed-form-substituted, so no labour axis is needed).
#   `Entry`     — EntryStage adds the entrant inflow `M·ν` (forward `Λ += g`; identity on V).
#
# The FORWARD sweep: entrants are seeded (Entry), survive-or-exit by the seated stopping
# rule (Exit drops low-z firms' mass — incl. entrants that draw too low a z), and the
# survivors' productivity transitions (Shock). Mass is NOT conserved (entry in, exit out);
# the stationary firm mass settles at entrant-inflow / exit-rate.
#
# OUTER LOOP (the caller's, never the block — exactly as for Aiyagari): the free-entry
# condition `∑_z ν(z)·V(z) = c_e` pins the equilibrium wage/price, and aggregate market
# clearing pins the entrant mass `M`. Both are scalar fixed points provided in
# `steady_state.jl` as plain driver arithmetic; here we solve the firm block at given
# prices (partial equilibrium).
#
# Literature: Hopenhayn (1992 Econometrica); Hopenhayn–Rogerson (1993) adds a firing
# cost (an ArgmaxStage(:n over {keep,adjust}) — a one-stage extension, see README).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct HopenhaynParams
    θ     :: Float64 = 0.64                  # span-of-control / decreasing returns to labour
    w     :: Float64 = 1.0                   # wage (the price the free-entry outer loop pins)
    c_f   :: Float64 = 0.40                  # per-period fixed operating cost
    scrap :: Float64 = 0.0                   # liquidation / scrap value (the exit composite's `bequest`)
    r     :: Float64 = 0.05                  # discount rate ⇒ β = 1/(1+r)
    M     :: Float64 = 1.0                   # entrant mass per period (pinned by clearing in GE)
    # Idiosyncratic productivity z: AR(1) in logs, mean ≈ 1.
    ρ_z   :: Float64 = 0.90
    σ_z   :: Float64 = 0.20
    N_z   :: Int     = 9
end

Base.Broadcast.broadcastable(p::HopenhaynParams) = Ref(p)

const hopenhayn_params = HopenhaynParams()


# Productivity process — Rouwenhorst discretization of the AR(1) in logs #
#-----------------------------------------------------------------------#

"""
Rouwenhorst discretization of an AR(1) `x' = ρ x + σ ε` into `n` states. Returns
`(grid, P)` with `grid` evenly spaced (half-width `σ·√((n−1)/(1−ρ²))`) and `P` the
`n×n` ROW-stochastic transition. Accurate at high persistence (unlike Tauchen).
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
        P[2:m-1, :]     ./= 2
    end
    ψ    = σ * sqrt((n - 1) / (1 - ρ^2))
    grid = collect(range(-ψ, ψ; length = n))
    return (grid, P)
end

"""
Invariant distribution of a row-stochastic transition `P` (left eigenvector at
eigenvalue 1), by power iteration. Used as the entrant productivity draw `ν`
(entrants look like a representative cross-section unless reweighted).
"""
function invariant_dist(P::AbstractMatrix; iters = 5000)
    n = size(P, 1)
    π = fill(1 / n, n)
    for _ in 1:iters
        π = vec(π' * P)
    end
    return π ./ sum(π)
end


# Per-period operating profit (static labour FOC, closed form) #
#--------------------------------------------------------------#

"""
Per-period operating profit `π(z) = max_n {z·n^θ − w·n} − c_f`. The static labour
choice is closed-form: `n*(z) = (θ z / w)^{1/(1−θ)}`, so no labour axis is needed —
the only firm state is productivity `z` (the standard Hopenhayn reduction).
"""
function operating_profit(z, p = hopenhayn_params)
    n_star = (p.θ * z / p.w)^(1 / (1 - p.θ))
    return z * n_star^p.θ - p.w * n_star - p.c_f
end


# Firm block assembly — FIVE library stages, NO bespoke stage #
#-------------------------------------------------------------#

"""
Build the Hopenhayn firm block
`Entry ∘ Profit ∘ Exit ∘ Discount ∘ Shock` with the firm mass, exit rate, mean
productivity, and mean (closed-form) employment attached as moments. The layout
carries the transient `:exiting` axis at size 1 that the exit composite grows to 2
and collapses back. Five existing stages, no bespoke firm stage.
"""
function hopenhayn_firm(p = hopenhayn_params)
    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid     = exp.(log_z)
    ν          = invariant_dist(P_z)               # entrant productivity distribution

    layout = GriddedLayout(
        :z       => Discrete(z_grid),
        :exiting => Discrete([0]),                  # transient axis the exit composite grows to 2
    )

    shock  = MarkovStage(layout; axis = :z, transition_matrix = P_z)
    profit = UtilityStage(layout; utility = (; z) -> operating_profit(z, p))
    disc   = TimeDiscountingStage(layout; β = 1 / (1 + p.r))
    exit   = EndogenousExit(layout; bequest = p.scrap)                 # max(continuation, scrap)
    entry  = EntryStage(layout; entry = reshape(p.M .* ν, p.N_z, 1))   # Λ += M·ν, shaped (z, exiting=1)

    firm = entry ∘ profit ∘ exit ∘ disc ∘ shock
    return define_moments!(firm;
        mass     = at_end(integrand = 1.0,  reduce = sum),
        mean_z   = at_end(integrand = :z,   reduce = sum),
        mean_emp = at_end(integrand = (; z) -> (p.θ * z / p.w)^(1 / (1 - p.θ)), reduce = sum),
    )
end


# Exogenous prices (plain function) + the free-entry residual (outer-loop arithmetic) #
#------------------------------------------------------------------------------------#

"Exogenous env: the discount rate (β baked into the stage); the wage rides `p.w`."
hopenhayn_env(p = hopenhayn_params) = (; p.r)

"""
Free-entry residual `(∑_z ν(z)·V(z)) − c_e`: in GE, entrants pay an entry cost `c_e`
and enter until the expected entrant value equals it; the root in the wage `w` is the
equilibrium price. PURE outer-loop arithmetic over the solved firm value `V` — NOT a
firm-block object (the block is solved at a given `w`). Returns the residual given the
entrant distribution `ν`, the solved value `V_z` over the z-grid, and the entry cost.
"""
free_entry_residual(V_z::AbstractVector, ν::AbstractVector, c_e::Real) = sum(ν .* V_z) - c_e
