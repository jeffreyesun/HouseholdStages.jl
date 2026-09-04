####################################################################################
# Clementi–Palazzo (2016) — entry + convex investment + endogenous exit, COMBINED     #
####################################################################################
#
# This example COMBINES two already-validated blocks into one firm recursion:
#   • the convex-investment block of `examples/capital_investment`
#     (`MarkovStage(:z) ∘ UtilityStage(z·k^α) ∘ CapitalInvestmentStage(:k)`), and
#   • the entry/exit composite of `examples/hopenhayn`
#     (`EntryStage(g) ∘ … ∘ EndogenousExit(scrap)` on a layout carrying `:exiting`).
# Clementi–Palazzo's point is that the INTERACTION amplifies fluctuations: entrants
# draw productivity and start small, invest toward their efficient scale subject to a
# convex adjustment cost, and exit endogenously when continuation falls below scrap.
#
# Household ↔ firm dictionary used here (§6 of EXAMPLES.md):
#   wealth `b`              ↔ capital `k`                  (the operative continuous axis)
#   income / employment shock ↔ productivity `z`          (MarkovStage)
#   saving                 ↔ physical investment `i = k'−(1−δ)k` (CapitalInvestmentStage)
#   convex saving/effort cost ↔ convex capital-adjustment cost φ·i² (its `effort_cost`)
#   death                  ↔ firm exit                    (EndogenousExit)
#   bequest / value of death ↔ scrap / liquidation value  (the REQUIRED `bequest` field)
#   birth                  ↔ firm entry                   (EntryStage, the additive Λ += g)
#
# A firm is an AGENT under this relabelling — nothing in the V/Λ machinery knows
# household from firm. The within-period firm block is EXISTING library stages, in
# time order, with NO bespoke stage:
#
#     Entry ∘ Profit ∘ Exit ∘ Invest ∘ Shock
#   = EntryStage(g) ∘ UtilityStage(z·k^α − c_f) ∘ EndogenousExit(scrap)
#       ∘ CapitalInvestmentStage(:k) ∘ MarkovStage(:z)
#
# The backward (value) sweep runs the chain right-to-left, reproducing the recursion:
#
#     V(k,z) = z·k^α − c_f
#              + max{ scrap(k) ,
#                     max_{k'}[ −φ·max(k'−(1−δ)k,0)² + β·E_{z'|z} V(k',z') ] }.
#
#   `Shock`   — MarkovStage backward forms the continuation expectation E[V(k',z')|z].
#   `Invest`  — CapitalInvestmentStage on `:k` (= ArgmaxStage ∘ TimeDiscountingStage):
#               supplies the discount β = 1/(1+r) and picks next capital k', paying the
#               convex cost φ·i² on gross investment i = k'−(1−δ)k (disinvestment free).
#               Production in this stage is 0 — operating profit lives in the UtilityStage
#               because the investment stage's `(value;env)` closures cannot see `z`.
#   `Exit`    — EndogenousExit takes max(continuation, scrap): a firm exits, collecting the
#               scrap value scrap(k) = resale·k, when its best continuation (already optimized
#               over k') falls below scrap. This is the §5(i) keep-vs-stop ArgmaxStage the
#               exit composite wraps over the transient `:exiting` axis.
#   `Profit`  — UtilityStage adds the per-period operating profit z·k^α net of the fixed
#               operating cost c_f. The c_f is LOAD-BEARING for exit: without a fixed cost
#               low-productivity firms' continuation never falls below scrap and nobody exits.
#   `Entry`   — EntryStage adds the entrant inflow M·g (forward Λ += g; identity on V).
#               Entrants start at the lowest capital grid point with z drawn from the
#               invariant distribution ν of the productivity chain — small and average-z.
#
# The FORWARD sweep: entrants are seeded small (Entry), this period's incumbents (incl.
# entrants) earn profit, the seated stopping rule drops low-(k,z) firms' mass (Exit), the
# survivors invest toward their target capital (Invest), and productivity transitions
# (Shock). Mass is NOT conserved (entry in, exit out); the stationary firm mass settles at
# entrant-inflow / exit-rate. Survivor selection: culling low-z firms lifts the mass-weighted
# mean z above the entrant mean.
#
# OUTER LOOP (the caller's, never the block — exactly as for Aiyagari): the free-entry
# condition `∑ g·V = c_e` pins the equilibrium price level, aggregate clearing pins the
# entrant mass `M`, and the firm-size (capital) distribution is an aggregate state. All are
# scalar/array fixed points the driver would run; here we solve the firm block at given
# prices (partial equilibrium) and expose `free_entry_residual` as the scalar to root on.
#
# Literature: Clementi–Palazzo (2016 AEJ:Macro); builds on Hopenhayn (1992) entry/exit and
# the Cooper–Haltiwanger (2006) convex adjustment cost.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct ClementiPalazzoParams
    α      :: Float64 = 0.70                  # curvature of operating profit z·k^α (<1: decreasing returns)
    φ      :: Float64 = 1.0                   # convex adjustment-cost coefficient: cost = φ·i²
    δ      :: Float64 = 0.15                  # capital depreciation
    r      :: Float64 = 0.10                  # discount rate ⇒ β = 1/(1+r) (high enough to limit the option value of riding out a bad streak)
    c_f    :: Float64 = 3.00                  # per-period FIXED operating cost (what makes low-z firms exit)
    resale :: Float64 = 0.60                  # scrap recovery rate: liquidation value scrap(k) = resale·k
    M      :: Float64 = 1.0                   # entrant mass per period (pinned by clearing in GE)
    # Idiosyncratic productivity z: AR(1) in logs, log z' = ρ log z + σ ε.
    # High persistence + wide dispersion: a low draw is a long, deep loss-making trap, so the
    # bottom productivity states' continuation falls below scrap and those firms exit.
    ρ_z    :: Float64 = 0.95                  # persistence of log productivity
    σ_z    :: Float64 = 0.40                  # innovation std of log productivity
    N_z    :: Int     = 7                     # number of productivity states
    N_k    :: Int     = 120                   # capital grid points
    k_min  :: Float64 = 0.30
    k_max  :: Float64 = 24.0
end

Base.Broadcast.broadcastable(p::ClementiPalazzoParams) = Ref(p)

const clementi_palazzo_params = ClementiPalazzoParams()


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
        P[2:m-1, :]     ./= 2                 # interior rows double-counted
    end
    ψ    = σ * sqrt((n - 1) / (1 - ρ^2))      # half-width of the state space
    grid = collect(range(-ψ, ψ; length = n))
    return (grid, P)
end

"""
Invariant distribution of a row-stochastic transition `P` (left eigenvector at
eigenvalue 1), by power iteration. Used as the entrant productivity draw `ν`:
entrants look like a representative productivity cross-section.
"""
function invariant_dist(P::AbstractMatrix; iters = 5000)
    n = size(P, 1)
    π = fill(1 / n, n)
    for _ in 1:iters
        π = vec(π' * P)
    end
    return π ./ sum(π)
end


# Entrant distribution — small capital, invariant productivity draw #
#------------------------------------------------------------------#

"""
Entrant distribution `g` over the full layout `(k, z, exiting=1)`: all entrant mass at
the lowest capital grid point (entrants start small) with productivity drawn from the
invariant distribution `ν` of `P_z`. Returns the `(N_k, N_z, 1)` array (un-scaled by the
entrant mass `M`, which `EntryStage` multiplies in).
"""
function entrant_distribution(ν::AbstractVector, N_k::Integer)
    N_z = length(ν)
    g = zeros(N_k, N_z, 1)
    g[1, :, 1] .= ν                            # lowest k; productivity ~ invariant dist
    return g
end


# Firm block assembly — FIVE library stages, NO bespoke stage #
#-------------------------------------------------------------#

"""
Build the Clementi–Palazzo firm block
`Entry ∘ Profit ∘ Exit ∘ Invest ∘ Shock` with firm mass, mean capital, mean productivity,
and mean profit attached as moments. The layout carries the transient `:exiting` axis at
size 1 that the exit composite grows to 2 and collapses back. Five existing stages, no
bespoke firm stage: the convex-investment block of `capital_investment` fused with the
entry/exit composite of `hopenhayn`.
"""
function clementi_palazzo_firm(p = clementi_palazzo_params)
    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid     = exp.(log_z)
    ν          = invariant_dist(P_z)               # entrant productivity distribution
    g          = entrant_distribution(ν, p.N_k)

    layout = GriddedLayout(
        :k       => GriddedContinuous(p.k_min, p.k_max, p.N_k),
        :z       => Discrete(z_grid),
        :exiting => Discrete([0]),                  # transient axis the exit composite grows to 2
    )

    shock  = MarkovStage(layout; axis = :z, transition_matrix = P_z)
    invest = CapitalInvestmentStage(layout;
        axis         = :k,
        β            = 1 / (1 + p.r),
        depreciation = p.δ,
        production   = (k) -> 0.0,                 # operating profit lives in the UtilityStage
        effort_cost  = (i) -> p.φ * i^2)           # convex cost on gross investment i ≥ 0
    exit   = EndogenousExit(layout; bequest = (; k) -> p.resale * k)   # max(continuation, resale·k)
    profit = UtilityStage(layout; utility = (; k, z) -> z * k^p.α - p.c_f)  # reads BOTH k and z
    entry  = EntryStage(layout; entry = p.M .* g)       # Λ += M·g, shaped (k, z, exiting=1)

    firm = entry ∘ profit ∘ exit ∘ invest ∘ shock
    return define_moments!(firm;
        mass        = at_end(integrand = 1.0,                reduce = sum),
        mean_k      = at_end(integrand = :k,                 reduce = sum),
        mean_z      = at_end(integrand = :z,                 reduce = sum),
        mean_profit = at_end(integrand = (; k, z) -> z * k^p.α, reduce = sum),
    )
end


# Exogenous prices (plain function) + the free-entry residual (outer-loop arithmetic) #
#------------------------------------------------------------------------------------#

"Exogenous env: the discount rate (β baked into the investment stage)."
clementi_palazzo_env(p = clementi_palazzo_params) = (; p.r)

"""
Free-entry residual `(∑_{k,z} g(k,z)·V(k,z)) − c_e`: in GE, entrants pay an entry cost
`c_e` and enter until the expected entrant value equals it; the root in the price level is
the equilibrium. PURE outer-loop arithmetic over the solved firm value `V` and the entrant
distribution `g` — NOT a firm-block object (the block is solved at given prices). `V` and
`g` are layout-shaped arrays `(N_k, N_z, 1)`.
"""
free_entry_residual(V::AbstractArray, g::AbstractArray, c_e::Real) = sum(g .* V) - c_e
