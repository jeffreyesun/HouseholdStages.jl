################################################################
# Lumpy / (S,s) labor adjustment (Cooper–Haltiwanger–Willis 2007) #
################################################################
#
# Household↔firm dictionary used here (§6 of MODEL_CATALOG.md):
#   wealth `b`        ↔ employment level `n`            (the operative axis)
#   income shock      ↔ productivity shock `z`          (MarkovStage)
#   (S,s) durable buy ↔ lumpy / fixed-cost hiring-firing (keep/adjust ArgmaxStage, §5(i))
#
# A FIXED cost of changing the employment level makes labor adjustment LUMPY: a firm sits in
# an inaction band and re-tunes its workforce only when productivity has drifted far enough to
# justify paying the cost. So hiring/firing is bursty and most firms sit still in any period.
# This is the §5(i) keep-vs-adjust shape — the SAME object as the (S,s) durable in
# `examples/durable_housing` and the lumpy capital in `examples/lumpy_investment`, read on the
# LABOR axis. The within-period firm block is EXISTING library stages, in time order, with NO
# bespoke stage:
#
#     Shock ∘ Profit ∘ Adjust ∘ Discount
#   = MarkovStage(:z) ∘ UtilityStage(z·n^θ − w·n) ∘ ArgmaxStage(:n; reward M[n',n]) ∘ TimeDiscountingStage(β)
#
# `Shock`    — MarkovStage on the productivity axis `:z` (Rouwenhorst AR(1) in logs).
# `Profit`   — UtilityStage adding flow profit NET of the wage bill: revenue `z·n^θ` (decreasing
#              returns / span of control, θ<1) minus the per-period wage bill `w·n`. Reads BOTH
#              axes (n and z); the adjustment ArgmaxStage's reward sees only the employment pair
#              (n', n), so the z-dependence of the flow MUST live in a separate UtilityStage —
#              exactly as in `examples/lumpy_investment`. The wage bill `w·n` is in the flow, so
#              HOLDING a level of employment is costly every period; only CHANGING the level pays F.
# `Adjust`   — ArgmaxStage on the discrete employment axis `:n` with the (S,s) reward
#              `M[n', n] = −F·1{n' ≠ n}`: keeping the headcount (n' = n, the diagonal) is free;
#              moving to any other level pays the FIXED hiring/firing cost F (the lump). The
#              inaction band is the set of (n, z) where re-tuning does not beat keeping.
#              `search = :brute`: the fixed cost makes the reward NON-supermodular, so the
#              monotone solve does not apply. A plain (after, before) Matrix IS the normal
#              ArgmaxStage reward parameterization.
# `Discount` — TimeDiscountingStage, β = 1/(1+r), supplying β·V_end before the argmax.
#
# The backward sweep reproduces the lumpy-labor Bellman
#     V(n,z) = z·n^θ − w·n + max_{n'} [ −F·1{n'≠n} + β·E[V(n',z')|z] ].
# Frictionless target (the level a firm would hold absent F): n*(z) = (θz/w)^{1/(1−θ)}.
#
# OUTER LOOP (the caller's): the wage `w` and the cross-sectional employment distribution as an
# aggregate state are partial-equilibrium-exogenous here (a single stationary solve), exactly as
# for Aiyagari. Closing labor-market clearing for `w` would be an outer fixed point on top.
#
# Literature: Cooper–Haltiwanger–Willis (2007, JME) — lumpy labor adjustment and the (S,s)
# employment band; Caballero–Engel–Haltiwanger (1997) on adjustment hazards. The same
# fixed-cost (S,s) object on the capital axis is `examples/lumpy_investment`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct LumpyLaborParams
    θ     :: Float64 = 0.64                   # curvature of revenue z·n^θ (span-of-control DRS)
    w     :: Float64 = 0.50                   # wage (per-period cost of holding the level n)
    F     :: Float64 = 0.20                   # FIXED cost of changing the employment level (the lump)
    r     :: Float64 = 0.04                   # discount rate ⇒ β = 1/(1+r)
    ρ_z   :: Float64 = 0.90
    σ_z   :: Float64 = 0.10
    N_z   :: Int     = 7
    N_n   :: Int     = 80                     # employment grid (DISCRETE: keep = stay at same index)
    n_min :: Float64 = 0.15
    n_max :: Float64 = 13.0
end

Base.Broadcast.broadcastable(p::LumpyLaborParams) = Ref(p)

const lumpy_labor_params = LumpyLaborParams()


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

"""
Frictionless employment target `n*(z) = (θz/w)^{1/(1−θ)}` — the level a firm would hold each
period absent the fixed cost. Used to size the employment grid so targets stay interior.
"""
n_star(z, p = lumpy_labor_params) = (p.θ * z / p.w)^(1 / (1 - p.θ))


# Firm block assembly — FOUR library stages, NO bespoke stage #
#-------------------------------------------------------------#

"""
Build the lumpy-labor firm block
`MarkovStage(:z) ∘ UtilityStage(z·n^θ − w·n) ∘ ArgmaxStage(:n; (S,s) reward) ∘ TimeDiscountingStage(β)`,
with mean employment, mean profit, and mean productivity attached. The employment axis is
DISCRETE so "keep" (n' = n) is an exact grid point. Four existing stages, no bespoke firm stage.
"""
function lumpy_labor_firm(p = lumpy_labor_params)
    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid     = exp.(log_z)
    n_grid     = collect(range(p.n_min, p.n_max; length = p.N_n))

    layout = GriddedLayout(
        :n => Discrete(n_grid),
        :z => Discrete(z_grid),
    )

    shock  = MarkovStage(layout; axis = :z, transition_matrix = P_z)
    # Flow profit: revenue z·n^θ (decreasing returns, θ<1) NET of the per-period wage bill w·n.
    # Reads BOTH axes (n and z); the adjustment ArgmaxStage's reward sees only the (n', n) pair, so
    # the z-dependence MUST live in a separate UtilityStage — as in `examples/lumpy_investment`.
    profit = UtilityStage(layout; utility = (; n, z) -> z * n^p.θ - p.w * n)

    # (S,s) adjustment reward matrix on the employment pair, M[n'(after), n(before)]: keeping the
    # headcount (n' = n, the diagonal) costs nothing; moving to any other level pays the FIXED
    # hiring/firing cost F (the lump). The per-period cost of the level itself is the wage bill in
    # `profit` above, so the only adjustment friction here is F — the textbook (S,s) inaction band.
    # A plain (after, before) Matrix IS the normal ArgmaxStage reward parameterization; the fixed
    # cost makes M non-supermodular ⇒ :brute.
    M = [jn == in ? 0.0 : -p.F for jn in 1:p.N_n, in in 1:p.N_n]        # M[after, before]
    adjust = ArgmaxStage(layout; reward = M, axis = :n, search = :brute) ∘
             TimeDiscountingStage(layout; β = 1 / (1 + p.r))

    firm = shock ∘ profit ∘ adjust
    return define_moments!(firm;
        mean_n      = at_end(integrand = :n, reduce = sum),
        mean_profit = at_end(integrand = (; n, z) -> z * n^p.θ - p.w * n, reduce = sum),
        mean_z      = at_end(integrand = :z, reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"Exogenous env: the discount rate only (β and w baked into the stages; no market to clear)."
lumpy_labor_env(p = lumpy_labor_params) = (; p.r)
