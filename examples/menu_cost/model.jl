################################################################
# Menu-cost price setting (Golosov–Lucas 2007; Nakamura–Steinsson 2010) — (S,s) on p #
################################################################
#
# Firm↔household dictionary (the (S,s) durable read on a different axis — §6 of EXAMPLES.md):
#   wealth `b` / capital `k` ↔ the firm's RELATIVE (real) price `p`   (the operative DISCRETE axis)
#   income / productivity shock ↔ idiosyncratic MARGINAL-COST shock `z` (MarkovStage, Rouwenhorst)
#   (S,s) durable buy / lumpy invest ↔ menu-cost price RESET           (keep/reset ArgmaxStage, §5(i))
#   depreciation / a deterministic WealthChange DRIFT ↔ INFLATION EROSION of the relative price
#
# A firm posts a NOMINAL price; with aggregate inflation π the price level rises each period, so a firm
# that does NOTHING sees its RELATIVE price erode downward. Re-pricing pays a fixed MENU COST `F`. The
# firm therefore sits in an inaction band — it lets its relative price drift down with inflation and
# only RESETS (jumps back up, paying `F`) once the gap to its frictionless optimum is large enough. This
# is the SAME keep-vs-adjust object as the lumpy-investment (S,s) capital band in `examples/lumpy_investment`,
# read on the relative-price axis instead of the capital axis. The within-period firm block is EXISTING
# library stages, in time order, with NO bespoke stage:
#
#     Shock ∘ Erosion ∘ Profit ∘ Reset ∘ Discount
#   = MarkovStage(:z) ∘ MarkovStage(:price; down-shift) ∘ UtilityStage(profit(p,z)) ∘ ArgmaxStage(:price; (S,s) reward) ∘ TimeDiscountingStage(β)
#
# `Shock`    — MarkovStage on the marginal-cost axis `:z` (Rouwenhorst AR(1)); z is the firm's
#              idiosyncratic marginal cost, the analogue of the income/productivity shock.
# `Erosion`  — MarkovStage on the DISCRETE `:price` axis whose transition is a DETERMINISTIC down-shift
#              permutation `T[i, max(i−1,1)] = 1`: a firm that keeps its nominal price slides DOWN exactly
#              one log-price grid step each period (the grid spacing IS the inflation step `log(1+π)`, so
#              erosion is exactly one index). The bottom index is absorbing. This is the inflation drift —
#              a normal MarkovStage parameterization (a 0/1 shift matrix), playing the role depreciation /
#              a deterministic WealthChange plays for a stock. It sits to the LEFT of the choice so, on the
#              backward sweep, the firm chooses AFTER the cost shock is known and the recursion already
#              accounts for next period's erosion.
# `Profit`   — UtilityStage adding monopolistic flow profit `profit(p,z) = D·(p^{1−ε} − z·p^{−ε})` with CES
#              demand elasticity ε: revenue `D·p^{1−ε}` minus cost `D·z·p^{−ε}`. Single-peaked in p with
#              frictionless optimum (the static markup) `p*(z) = (ε/(ε−1))·z`. Reads BOTH axes; the reset
#              ArgmaxStage's reward sees only the (p', p) pair, so the profit MUST live in its own
#              UtilityStage — exactly as the operating profit does in `examples/lumpy_investment`.
# `Reset`    — ArgmaxStage on the discrete `:price` axis with the (S,s) reward `M[p', p] = −F·1{p' ≠ p}`:
#              KEEPING the price (p' = p, the diagonal) is free; RESETTING to any other grid price pays the
#              fixed MENU COST `F`. The continuation value `β·V` (single-peaked in p', inherited from the
#              profit) supplies the benefit of moving; `F` is the only friction here. The band is the set of
#              (p, z) where resetting does not beat keeping. Brute argmax: the fixed cost makes the reward
#              NON-supermodular (a monotone walk would mis-solve). A plain (after, before)
#              `Matrix` IS the normal ArgmaxStage reward parameterization.
# `Discount` — TimeDiscountingStage, β = 1/(1+r), supplying β·V_end before the argmax.
#
# The backward sweep reproduces the menu-cost Bellman
#     V(p,z) = E_{z'|z}[ profit(p_e, z') + max_{p'} ( −F·1{p'≠p_e} + β·V(p', z') ) ],   p_e = erode(p) = one step down.
#
# OUTER LOOP (the caller's, NOT modeled here): the MONETARY block — the aggregate price level, the
# inflation rate π, and the cross-sectional price distribution as an aggregate state — is treated as
# partial-equilibrium-exogenous (a single stationary solve at a fixed π), exactly as the rental rate is
# for lumpy investment and the interest rate is for Aiyagari. Closing the model (Calvo-vs-menu Phillips
# curve, monetary policy) would wrap this block in a fixed point on π and the price level.
#
# Literature: Sheshinski–Weiss (1977) the (S,s) pricing band; Golosov–Lucas (2007 JPE); Nakamura–Steinsson
# (2010 QJE) the frequency-of-price-change moment; Barro (1972) the band-width / menu-cost balance.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct MenuCostParams
    ε       :: Float64 = 4.0                 # CES demand elasticity ⇒ frictionless markup ε/(ε−1) = 4/3
    D       :: Float64 = 1.0                 # demand scale (level of profit)
    F       :: Float64 = 0.025               # fixed MENU COST of resetting the price (the lump)
    r       :: Float64 = 0.04                # discount rate ⇒ β = 1/(1+r)
    π       :: Float64 = 0.03                # aggregate inflation: one erosion step per period = log(1+π)
    ρ_z     :: Float64 = 0.90                # persistence of the marginal-cost shock
    σ_z     :: Float64 = 0.03                # std of the marginal-cost innovation
    N_z     :: Int     = 5
    N_p     :: Int     = 30                  # relative-price grid (DISCRETE: keep = stay at same index)
    p_lo    :: Float64 = 0.85                # bottom of the log-price grid (below the band, room to drift)
end

Base.Broadcast.broadcastable(p::MenuCostParams) = Ref(p)

const menu_cost_params = MenuCostParams()


# Marginal-cost process — Rouwenhorst #
#------------------------------------#

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


# The (log-)price grid and the inflation-erosion shift #
#------------------------------------------------------#

"""
Log-spaced relative-price grid whose step equals the inflation step `log(1+π)`, so a one-period
erosion is exactly a one-index down-shift. Starts at `p_lo` and rises `N_p` points.
"""
function price_grid(p::MenuCostParams)
    Δ = log(1 + p.π)
    return exp.(log(p.p_lo) .+ Δ .* (0:p.N_p-1))
end

"""
Deterministic inflation-erosion transition on the price axis: row `i` (a firm currently at price
index `i`) sends all its mass DOWN one step to `max(i−1, 1)`. Row-stochastic; index 1 is absorbing.
This is the inflation drift, expressed as a normal MarkovStage transition (a 0/1 shift permutation).
"""
function erosion_matrix(N_p::Integer)
    T = zeros(N_p, N_p)
    for i in 1:N_p
        T[i, max(i - 1, 1)] = 1.0
    end
    return T
end

"""
Monopolistic flow profit at relative price `p` and marginal cost `z` under CES demand with
elasticity `ε` and demand scale `D`: revenue `D·p^{1−ε}` minus cost `D·z·p^{−ε}`. Single-peaked
in `p` with frictionless optimum `p*(z) = (ε/(ε−1))·z`.
"""
menu_cost_profit(p, z, par::MenuCostParams) = par.D * (p^(1 - par.ε) - z * p^(-par.ε))


# Firm block assembly — FIVE library stages, NO bespoke stage #
#-------------------------------------------------------------#

"""
Build the menu-cost firm block
`MarkovStage(:z) ∘ MarkovStage(:price; down-shift) ∘ UtilityStage(profit) ∘ ArgmaxStage(:price; (S,s) reward) ∘ TimeDiscountingStage(β)`,
with mean price, mean profit, and mean marginal cost attached. The price axis is DISCRETE so
"keep" (p' = p) is an exact grid point and the inflation erosion is an exact one-index shift.
Five existing stages, no bespoke firm stage.
"""
function menu_cost_firm(p = menu_cost_params)
    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid     = exp.(log_z)
    p_grid     = price_grid(p)

    layout = GriddedLayout(
        :price => Discrete(p_grid),
        :z     => Discrete(z_grid),
    )

    shock   = MarkovStage(layout; axis = :z,     transition_matrix = P_z)
    # Inflation erosion: deterministic down-shift on the price axis. The grid step IS log(1+π), so a
    # firm that keeps its nominal price slides down exactly one index each period (the relative-price drift).
    erosion = MarkovStage(layout; axis = :price, transition_matrix = erosion_matrix(p.N_p))
    # Monopolistic flow profit. Reads BOTH axes (price and z); the reset ArgmaxStage's reward sees only
    # the (p', p) pair, so the z-dependence of the flow MUST live in a separate UtilityStage.
    profit  = UtilityStage(layout; utility = (; price, z) -> menu_cost_profit(price, z, p))

    # (S,s) reset reward on the price pair, M[p'(after), p(before)]: KEEPING the price (p' = p, the
    # diagonal) is free; RESETTING to any other grid price pays the fixed MENU COST F. A plain (after,
    # before) Matrix IS the normal ArgmaxStage reward parameterization; non-supermodular, so the
    # brute ArgmaxStage (not ContinuousArgmaxStage) is the right primitive.
    M = [jp == ip ? 0.0 : -p.F for jp in 1:p.N_p, ip in 1:p.N_p]      # M[after, before]
    reset = ArgmaxStage(layout; reward = M, axis = :price) ∘
            TimeDiscountingStage(layout; β = 1 / (1 + p.r))

    firm = shock ∘ erosion ∘ profit ∘ reset
    return define_moments!(firm;
        mean_price  = at_end(integrand = :price, reduce = sum),
        mean_profit = at_end(integrand = (; price, z) -> menu_cost_profit(price, z, p), reduce = sum),
        mean_z      = at_end(integrand = :z, reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"Exogenous env: the discount rate and inflation only (both baked into the stages; no market to clear)."
menu_cost_env(p = menu_cost_params) = (; p.r, p.π)
