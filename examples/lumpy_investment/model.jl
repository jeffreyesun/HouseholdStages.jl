################################################################
# Lumpy / non-convex investment (Khan–Thomas 2008) — (S,s) on k  #
################################################################
#
# Household↔firm dictionary used here (§6 of EXAMPLES.md):
#   wealth `b`        ↔ capital `k`                   (the operative continuous axis)
#   income shock      ↔ productivity shock `z`        (MarkovStage)
#   (S,s) durable buy ↔ lumpy / fixed-cost investment (keep/adjust ArgmaxStage, §5(i))
#
# A fixed adjustment cost makes investment LUMPY: a firm sits in an inaction band and
# adjusts its capital level only when productivity has drifted far enough to justify the
# cost. This is the §5(i) keep-vs-adjust shape — the SAME object as the (S,s) durable in
# `examples/durable_housing`, read on the capital axis. The within-period firm block is
# EXISTING library stages, in time order, with NO bespoke stage:
#
#     Shock ∘ Profit ∘ Invest ∘ Discount
#   = MarkovStage(:z) ∘ UtilityStage(z·k^α − δ·k) ∘ ArgmaxStage(:k; reward M[k',k]) ∘ TimeDiscountingStage(β)
#
# `Shock`    — MarkovStage on the profitability axis `:z` (Rouwenhorst AR(1)).
# `Profit`   — UtilityStage adding operating profit NET of frictionless maintenance:
#              `z·k^α − δ·k`. Reads BOTH axes (k and z); the investment ArgmaxStage's
#              reward sees only the capital pair (k', k), so the z-dependence of the flow
#              MUST live in a separate UtilityStage — exactly as in `examples/capital_investment`.
#              Maintenance `δ·k` (replacing depreciation at the no-friction price 1) is in the
#              flow, so HOLDING a level is costly while a small re-tuning is free; only changing
#              the LEVEL pays the fixed cost.
# `Invest`   — ArgmaxStage on the discrete capital axis `:k` with the (S,s) reward
#              `M[k', k] = −(k' − k) − F·1{k' ≠ k}`: keeping the level (k' = k, the diagonal)
#              is free; moving to any other level pays the linear capital cost `(k' − k)` PLUS
#              the fixed cost `F`. The inaction band is the set of (k, z) where re-tuning does
#              not beat keeping. Brute argmax: the fixed cost makes the reward NON-supermodular
#              (a monotone walk would mis-solve; the continuous stage's guard would refuse it).
# `Discount` — TimeDiscountingStage, β = 1/(1+r), supplying β·V_end before the argmax.
#
# The backward sweep reproduces the lumpy-investment Bellman
#     V(k,z) = z·k^α − (r+δ)·k + max_{k'} [ −F·1{k'≠k} + β·E[V(k',z')|z] ].
#
# OUTER LOOP (the caller's): the rental rate / capital price and the cross-sectional capital
# distribution as an aggregate state are partial-equilibrium-exogenous here (a single
# stationary solve), exactly as for Aiyagari.
#
# Literature: Khan–Thomas (2008 Econometrica); Cooper–Haltiwanger (2006) fixed-cost
# component; Bachmann–Caballero–Engel (2013). The convex component is `examples/capital_investment`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct LumpyInvestmentParams
    α     :: Float64 = 0.30                  # curvature of operating profit z·k^α (span-of-control DRS)
    δ     :: Float64 = 0.10                  # depreciation (enters the per-period user cost)
    F     :: Float64 = 0.50                  # FIXED cost of changing the capital level (the lump)
    r     :: Float64 = 0.04                  # discount rate ⇒ β = 1/(1+r); user cost (r+δ)
    ρ_z   :: Float64 = 0.85
    σ_z   :: Float64 = 0.25
    N_z   :: Int     = 7
    N_k   :: Int     = 60                    # capital grid (DISCRETE: keep = stay at same index)
    k_min :: Float64 = 0.5
    k_max :: Float64 = 16.0
end

Base.Broadcast.broadcastable(p::LumpyInvestmentParams) = Ref(p)

const lumpy_investment_params = LumpyInvestmentParams()


# Productivity process — Rouwenhorst #
#-----------------------------------#

"""
Rouwenhorst discretization of `x' = ρ x + σ ε` into `n` states; returns `(grid, P)`
with `P` row-stochastic. Accurate at high persistence.
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


# Firm block assembly — FOUR library stages, NO bespoke stage #
#-------------------------------------------------------------#

"""
Build the lumpy-investment firm block
`MarkovStage(:z) ∘ UtilityStage(z·k^α − δ·k) ∘ ArgmaxStage(:k; (S,s) reward) ∘ TimeDiscountingStage(β)`,
with mean capital, mean profit, and the adjustment frequency (fraction of firms moving
their capital level) attached. The capital axis is DISCRETE so "keep" (k' = k) is an
exact grid point. Four existing stages, no bespoke firm stage.
"""
function lumpy_investment_firm(p = lumpy_investment_params)
    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid     = exp.(log_z)
    k_grid     = collect(range(p.k_min, p.k_max; length = p.N_k))

    layout = GriddedLayout(
        :k => Discrete(k_grid),
        :z => Discrete(z_grid),
    )

    shock  = MarkovStage(layout; axis = :z, transition_matrix = P_z)
    # Operating profit NET of the per-period Jorgensonian user cost (r+δ)·k of holding the level k.
    # Reads BOTH axes (k and z); the investment ArgmaxStage's reward sees only the (k', k) pair, so the
    # z-dependence MUST live in a separate UtilityStage — as in `examples/capital_investment`.
    profit = UtilityStage(layout; utility = (; k, z) -> z * k^p.α - (p.r + p.δ) * k)

    # (S,s) investment reward matrix on the capital pair, M[k'(after), k(before)]: keeping the level
    # (k' = k, the diagonal) costs nothing; jumping to any other level pays the FIXED cost F (the lump).
    # The per-period cost of the level itself is the user cost charged in `profit` above, so the only
    # adjustment friction here is F — the textbook (S,s) capital target/inaction structure. A plain
    # (after, before) Matrix IS the normal ArgmaxStage reward parameterization; non-supermodular,
    # so the brute ArgmaxStage (not ContinuousArgmaxStage) is the right primitive.
    M = [jk == ik ? 0.0 : -p.F for jk in 1:p.N_k, ik in 1:p.N_k]        # M[after, before]
    invest = ArgmaxStage(layout; reward = M, axis = :k) ∘
             TimeDiscountingStage(layout; β = 1 / (1 + p.r))

    firm = shock ∘ profit ∘ invest
    return define_moments!(firm;
        mean_k      = at_end(integrand = :k, reduce = sum),
        mean_profit = at_end(integrand = (; k, z) -> z * k^p.α, reduce = sum),
        mean_z      = at_end(integrand = :z, reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"Exogenous env: the discount rate only (β baked into the stage; no market to clear)."
lumpy_investment_env(p = lumpy_investment_params) = (; p.r)
