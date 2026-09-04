#########################################################################
# Tombe–Zhu (2019) — migration over a region×sector composite axis        #
#########################################################################

# "Trade, Migration, and Productivity: A Quantitative Analysis of China"
# (AER 2019): workers reallocate across REGION-SECTOR cells, paying a
# bilateral migration cost that depends on both the region change and the
# sector change. The defining feature for HouseholdStages is that ONE
# discrete choice moves a COMPOSITE axis whose levels are `(region, sector)`
# pairs — a single `MigrationStage` over that composite axis, no bespoke
# stage. The within-period problem is, in time order:
#
#     Migrate ∘ Receipt ∘ ConsumptionSavings
#
# Library stages (NO bespoke household stage in this file):
#   Migrate  — `MigrationStage` (sugar over `LogitChoiceStage`) on the
#              composite `:rs` axis whose levels are `(region, sector)`
#              tuples. The cost `M[(r,s) → (r',s')]` adds a region-move cost
#              and a sector-switch cost (zero diagonal); it is plain data.
#   Receipt  — `WealthChangeStage`: cash-on-hand `(1+r)·a + real_wage(r,s)`,
#              the region×sector real wage read from `env`.
#   ConsumptionSavings — `ConsumptionSavingsStage` on the wealth grid.
#
# The migration choice makes the composite axis ergodic. The real wages
# (which embed trade through goods prices) and the spatial measure clear in
# the trade-and-migration GE — the caller's OUTER loop. This file solves the
# household block at a fixed `env` (partial equilibrium).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct TombeZhuParams
    β :: Float64 = 0.96
    σ :: Float64 = 1.5
    r :: Float64 = 0.03
    regions :: Vector{Symbol} = [:coast, :interior]
    sectors :: Vector{Symbol} = [:ag, :mfg]
    # Real wage by (region, sector) — coast manufacturing pays most.
    #               (coast,ag) (coast,mfg) (interior,ag) (interior,mfg)
    real_wage :: Vector{Float64} = [0.9,      1.3,         0.8,           1.0]
    region_move_cost :: Float64 = 0.50   # utility cost to change region
    sector_move_cost :: Float64 = 0.30   # utility cost to change sector
    ε :: Float64 = 0.40
    N_w   :: Int     = 200
    w_min :: Float64 = 0.0
    w_max :: Float64 = 40.0
end

Base.Broadcast.broadcastable(p::TombeZhuParams) = Ref(p)

const params = TombeZhuParams()


# Composite axis + migration-cost matrix (plain primitives, not stages) #
#----------------------------------------------------------------------#

"""
The composite axis levels: every `(region, sector)` pair, in the order
`regions × sectors` (matching the `real_wage` vector). Plain layout data.
"""
rs_levels(p = params) = [(rg, sc) for rg in p.regions for sc in p.sectors]

"""
The `n_rs × n_rs` migration-cost matrix over the composite `(region, sector)`
axis: moving `(r,s) → (r',s')` costs `region_move_cost·1{r≠r'} +
sector_move_cost·1{s≠s'}` (zero on the diagonal). One discrete move spans
both margins. Plain data handed to `MigrationStage`.
"""
function rs_cost_matrix(p = params)
    lv = rs_levels(p)
    n  = length(lv)
    return [begin
                (r0, s0) = lv[i]; (r1, s1) = lv[j]
                p.region_move_cost * (r0 != r1) + p.sector_move_cost * (s0 != s1)
            end for i in 1:n, j in 1:n]
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Tombe–Zhu household block
`Migrate ∘ Receipt ∘ ConsumptionSavings` over (wealth, rs). The composite
`:rs` axis carries `(region, sector)` pairs; one `MigrationStage` reallocates
over both. Moments: aggregate wealth and population by composite cell.
"""
function tombe_zhu_household(p = params)
    lv = rs_levels(p)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :rs     => Discrete(lv),
    )

    migrate = MigrationStage(layout;
        axis           = :rs,
        migration_cost = rs_cost_matrix(p),
        ε              = p.ε)
    receipt = WealthChangeStage(layout;                          # defaults: (; axis = :wealth)
        wealth_post = function (; rs, wealth, env)
            k = findfirst(==(rs), env.rs_levels)
            return (1 + env.r) * wealth + env.real_wage[k]
        end)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)))

    hh = migrate ∘ receipt ∘ savings

    at_cell(cell) = (; rs) -> rs == cell ? 1.0 : 0.0
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
        pop_coast_ag  = at_end(integrand = at_cell((:coast, :ag)),     reduce = sum),
        pop_coast_mfg = at_end(integrand = at_cell((:coast, :mfg)),    reduce = sum),
        pop_int_ag    = at_end(integrand = at_cell((:interior, :ag)),  reduce = sum),
        pop_int_mfg   = at_end(integrand = at_cell((:interior, :mfg)), reduce = sum),
    )
end


# Env builder (plain function) #
#------------------------------#

"The env consumed by the chain: real wages + the composite levels (fixed)."
tombe_zhu_env(p = params) = (; r = p.r, real_wage = p.real_wage, rs_levels = rs_levels(p))
