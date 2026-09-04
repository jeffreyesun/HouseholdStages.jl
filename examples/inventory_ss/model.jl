################################################################
# (S,s) inventory management (Khan–Thomas 2007; Kahn–McConnell–   #
# Pérez-Quirós 2002) — (S,s) on a stock that DEPLETES each period #
################################################################
#
# Durable-buyer↔firm dictionary used here (the (S,s) inaction-band shape, §5(i)):
#   wealth / durable stock     ↔ inventory level `i`        (the operative DISCRETE axis)
#   income shock               ↔ demand shock `d`           (MarkovStage)
#   transaction / menu cost    ↔ FIXED reorder cost `F`     (keep/adjust ArgmaxStage)
#   — (NEW, vs the durable)    ↔ DEPLETION DRIFT            (sales draw the stock DOWN each period)
#
# A firm holds inventory, meets stochastic demand by selling out of stock, and pays a FIXED cost
# every time it places a reorder. The fixed cost makes reordering LUMPY: the firm lets its stock
# draw DOWN as sales accumulate, sitting in an inaction band, and reorders up to a target `S` only
# once the stock has fallen below a trigger `s` — the textbook (S,s) policy. This is the SAME
# keep-vs-adjust object as the (S,s) durable in `examples/durable_housing` and the (S,s) capital in
# `examples/lumpy_investment`, read on the INVENTORY axis, with ONE structural addition: between
# periods the stock is not inert — it is drawn down by realized sales (the DEPLETION DRIFT).
#
# The within-period firm block is EXISTING library stages, in time order, with NO bespoke stage:
#
#     Demand ∘ Profit ∘ Deplete ∘ Reorder ∘ Discount
#   = MarkovStage(:demand)                                   demand shock d ~ Markov
#       ∘ UtilityStage(price·sales − h·hold − κ·stockout)    flow profit, reads (i, d)
#       ∘ MarkovStage(:inventory; T = (; demand) -> shift)   DEPLETION: i ↦ max(i−d, 0)
#       ∘ ArgmaxStage(:inventory; reward M[i', s])           (S,s) reorder: keep s, or pay F + c·Δ
#       ∘ TimeDiscountingStage(β)                            β = 1/(1+r)
#
# `Demand`   — MarkovStage on the demand axis `:demand` (Rouwenhorst AR(1) on log demand).
# `Profit`   — UtilityStage adding the within-period flow profit. Reads BOTH the inventory level `i`
#              (the stock at the START of the period, before depletion) and the realized demand `d`:
#                  profit = price·sales − h·max(i−d, 0) − κ·max(d−i, 0),   sales = min(i, d).
#              Revenue on units sold, holding cost `h` on the stock left over after sales, stockout
#              penalty `κ` on demand that could not be met. It MUST read the pre-depletion `i`, so it
#              precedes the depletion move (which overwrites the inventory axis with the leftover).
# `Deplete`  — the DEPLETION DRIFT, expressed as a MarkovStage on the `:inventory` axis whose
#              transition DEPENDS on the demand state via a dep-closure `(; demand) -> shift(demand)`
#              (the same dep-closure-transition contract as `examples/skill_depreciation`'s
#              skill drift `(; emp) -> …` and `examples/marriage_capital`'s `(; match_capital) -> …`).
#              `shift(d)` is the deterministic ROW-stochastic map sending each stock index `i` to the
#              index nearest `max(grid[i] − d, 0)` — sales `min(i, d)` knock the stock down, floored at
#              empty. A deterministic move IS a one-hot Markov matrix, so this is an ordinary
#              MarkovStage, not a bespoke kernel.
# `Reorder`  — ArgmaxStage on the discrete inventory axis with the (S,s) reorder reward
#              `M[i'(after), s(before)]`: keeping the post-depletion stock (i' = s, the diagonal) is
#              free; reordering UP to any higher target pays the FIXED cost `F` PLUS the per-unit
#              purchase cost `c·(grid[i'] − grid[s])`; reducing the stock (i' < s) is infeasible
#              (`−Inf`). The inaction band is the set of post-depletion stocks at which keeping beats
#              every reorder. Brute argmax: the fixed cost makes the reward NON-supermodular (a
#              monotone walk would mis-solve; the continuous stage's guard would refuse it).
# `Discount` — TimeDiscountingStage, β = 1/(1+r), supplying β·V_end before the reorder argmax.
#
# Period timing (one firm): enter with stock `i` (last period's reorder target). Demand `d` realizes;
# the firm earns `profit(i, d)`; the stock depletes to `s = max(i − d, 0)`; the firm reorders up to
# `i' ≥ s` (paying `F + c·Δ`) or keeps `s`; discount; carry `i'` into next period. The backward sweep
# reproduces the (S,s) inventory Bellman
#     V(i, d_prev) = E_{d|d_prev}[ profit(i, d) + max_{i' ≥ s(i,d)} ( −F·1{i'≠s} − c·(i'−s) + β·V(i', d) ) ].
#
# OUTER LOOP (the caller's): the goods price, the per-unit purchase cost, and the cross-sectional
# inventory distribution as an aggregate state are partial-equilibrium-exogenous here (a single
# stationary solve), exactly as for Aiyagari / lumpy investment.
#
# Literature: Scarf (1960) (S,s) optimality; Khan–Thomas (2007 JPE) inventories in GE; Kahn,
# McConnell & Pérez-Quirós (2002) inventories and the Great Moderation. The non-depleting (S,s)
# stock is `examples/durable_housing`; the (S,s) capital is `examples/lumpy_investment`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct InventoryParams
    price :: Float64 = 1.00                  # sale price per unit (revenue = price · sales)
    c     :: Float64 = 0.55                  # per-unit purchase cost of a reorder (margin price − c > 0)
    h     :: Float64 = 0.05                  # per-unit holding cost on leftover stock after sales
    κ     :: Float64 = 0.80                  # per-unit stockout penalty on unmet demand
    F     :: Float64 = 0.60                  # FIXED cost of placing a reorder (the lump ⇒ (S,s))
    r     :: Float64 = 0.05                  # discount rate ⇒ β = 1/(1+r)

    ρ_d   :: Float64 = 0.50                  # demand persistence (Rouwenhorst AR(1) on log demand)
    σ_d   :: Float64 = 0.35
    N_d   :: Int     = 5                     # demand states
    d_bar :: Float64 = 1.00                  # demand scale (median demand level)

    N_i   :: Int     = 49                    # inventory grid (DISCRETE: keep = stay at same index)
    i_min :: Float64 = 0.0
    i_max :: Float64 = 12.0                  # S_max — above the reorder target
end

Base.Broadcast.broadcastable(p::InventoryParams) = Ref(p)

const inventory_params = InventoryParams()


# Demand process — Rouwenhorst #
#-----------------------------#

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


# Depletion drift — a deterministic ROW-stochastic move on the inventory axis #
#----------------------------------------------------------------------------#

"""
Index of the `i_grid` point nearest the value `x` (linear scan; `i_grid` is sorted).
"""
function nearest_index(i_grid::AbstractVector, x::Real)
    best_j = 1
    best_δ = abs(i_grid[1] - x)
    @inbounds for j in 2:length(i_grid)
        δ = abs(i_grid[j] - x)
        δ < best_δ && (best_δ = δ; best_j = j)
    end
    return best_j
end

"""
Depletion transition `shift(d)` on the inventory grid for realized demand `d`: a ROW-stochastic
ONE-HOT matrix `T[i_from, i_to]` sending each start stock `grid[i_from]` to the grid index nearest
`max(grid[i_from] − d, 0)` — sales `min(stock, d)` draw the stock down, floored at empty. Deterministic,
so each row has a single `1`; this is exactly the dep-closure matrix `MarkovStage(:inventory)` consumes
per demand state (the `(; demand) -> shift(demand)` contract, as in `examples/skill_depreciation`).
"""
function depletion_matrix(i_grid::AbstractVector, d::Real)
    n = length(i_grid)
    T = zeros(n, n)
    @inbounds for ii in 1:n
        T[ii, nearest_index(i_grid, max(i_grid[ii] - d, 0.0))] = 1.0
    end
    return T
end


# Firm block assembly — FIVE library stages, NO bespoke stage #
#-------------------------------------------------------------#

"""
Build the (S,s) inventory firm block
`MarkovStage(:demand) ∘ UtilityStage(profit) ∘ MarkovStage(:inventory; depletion) ∘ ArgmaxStage(:inventory; (S,s) reorder reward) ∘ TimeDiscountingStage(β)`,
with mean inventory, mean sales, and mean profit attached. The inventory axis is DISCRETE so "keep"
(i' = s) is an exact grid point, and the depletion drift is a dep-closure MarkovStage on `:inventory`
reading the `:demand` value. Five existing stages, no bespoke firm stage.
"""
function inventory_firm(p = inventory_params)
    log_d, P_d = rouwenhorst(p.ρ_d, p.σ_d, p.N_d)
    d_grid     = p.d_bar .* exp.(log_d)
    i_grid     = collect(range(p.i_min, p.i_max; length = p.N_i))

    layout = GriddedLayout(
        :inventory => Discrete(i_grid),
        :demand    => Discrete(d_grid),
    )

    demand = MarkovStage(layout; axis = :demand, transition_matrix = P_d)

    # Flow profit, reading the PRE-depletion stock `i` and the realized demand `d`:
    # revenue on units sold, holding cost on leftover, stockout penalty on unmet demand.
    profit = UtilityStage(layout; utility = (; inventory, demand) ->
        p.price * min(inventory, demand) -
        p.h * max(inventory - demand, 0.0) -
        p.κ * max(demand - inventory, 0.0))

    # DEPLETION DRIFT: a MarkovStage on `:inventory` whose ROW-stochastic transition DEPENDS on the
    # `:demand` value — the deterministic draw-down `i ↦ max(i − d, 0)` (the dep-closure contract).
    deplete = MarkovStage(layout; axis = :inventory,
        transition_matrix = (; demand) -> depletion_matrix(i_grid, demand))

    # (S,s) reorder reward on the inventory pair, M[i'(after), s(before)]: keeping the post-depletion
    # stock (i' = s, the diagonal) is free; reordering UP to a higher target pays the FIXED cost F PLUS
    # the per-unit purchase cost c·(grid[i'] − grid[s]); reducing the stock (i' < s) is infeasible.
    # A plain (after, before) Matrix IS the normal ArgmaxStage reward; non-supermodular, so the
    # brute ArgmaxStage (not ContinuousArgmaxStage) is the right primitive.
    M = [ji == ii ? 0.0 :
         ji  > ii ? -(p.F + p.c * (i_grid[ji] - i_grid[ii])) :
         -Inf
         for ji in 1:p.N_i, ii in 1:p.N_i]                              # M[after, before]
    reorder = ArgmaxStage(layout; reward = M, axis = :inventory) ∘
              TimeDiscountingStage(layout; β = 1 / (1 + p.r))

    firm = demand ∘ profit ∘ deplete ∘ reorder
    return define_moments!(firm;
        mean_inventory = at_end(integrand = :inventory, reduce = sum),
        mean_sales     = at_end(integrand = (; inventory, demand) -> min(inventory, demand), reduce = sum),
        mean_profit    = at_end(integrand = (; inventory, demand) ->
            p.price * min(inventory, demand) - p.h * max(inventory - demand, 0.0) -
            p.κ * max(demand - inventory, 0.0), reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"Exogenous env: the discount rate only (β baked into the stage; no market to clear)."
inventory_env(p = inventory_params) = (; p.r)
