#########################################################
# Roy-style Occupational Choice — SectorSwitchingStage   #
#########################################################

# A worker who each period chooses which sector (occupation) to work in.
# Sectors pay different wages per efficiency unit, idiosyncratic
# productivity follows a within-period Markov draw, and switching sectors
# costs a stage-specific utility penalty. The within-period problem
# decomposes into four stages, in time order:
#
#     ProductivityShock ∘ SectorSwitch ∘ Earnings ∘ ConsumptionSavings
#
# Library stages used (NO bespoke household stage in this file):
#   ProductivityShock — `MarkovStage` on the :prod axis (the idiosyncratic
#                        efficiency-unit draw, common across sectors).
#   SectorSwitch      — `SectorSwitchingStage` (sugar over LogitChoiceStage)
#                        on the :sector axis: logit choice of next sector
#                        with a `switching_cost[i, j]` utility penalty for
#                        moving i → j (free to stay, zero diagonal).
#   Earnings          — `WealthChangeStage` (deterministic): the chosen
#                        sector's wage `w[sector]` times productivity is
#                        received and added to (1+r) wealth.
#   ConsumptionSavings— `ConsumptionSavingsStage` on the wealth grid.
#
# Roy (1951) supplies the comparative-advantage logic: sectors differ in
# their wage per efficiency unit, so a worker's optimal sector depends on
# their idiosyncratic productivity draw; the switching cost (as in the
# occupational-mobility literature, e.g. Artuç-Chaudhuri-McLaren 2010,
# Dix-Carneiro 2014) makes the choice dynamic and history-dependent rather
# than a static cross-section sort.
#
# The sector choice is resolved on the SAME timeline as a spatial migration
# choice (cf. `../spatial/model.jl`): the worker sees their productivity
# draw, then chooses a sector (logit-smoothed, paying the switching cost),
# then receives the sector wage and chooses savings.
#
# The wealth grid is log-spaced (dense near zero where the borrowing
# constraint binds, coarse at the top so the post-earnings wealth stays
# in-grid). This is a partial-equilibrium / fixed-price stationary model:
# sector wages and the interest rate are exogenous parameters, and we solve
# the stationary distribution at a given env. (A market-clearing closure on
# the wages would be a plain outer loop, analogous to spatial's
# tatonnement; it is not needed to exercise the household block.)

using HouseholdStages


# Parameters #
#------------#

@kwdef struct SectoralParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    r :: Float64       = 0.03           # exogenous (partial-equilibrium) return on wealth
    # Idiosyncratic productivity (efficiency units), common across sectors.
    prod_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_prod    :: Matrix{Float64} = [0.7 0.2 0.1;
                                    0.2 0.6 0.2;
                                    0.1 0.2 0.7]
    # Sector wages per efficiency unit (Roy comparative advantage lives in
    # the interaction of these levels with the productivity draw + the
    # switching cost).
    sectors  :: Vector{Symbol}   = [:ag, :mfg, :svc]
    w_sector :: Vector{Float64}  = [0.9, 1.1, 1.0]
    # Switching cost (utils) for moving sector i → j; zero diagonal.
    κ :: Float64       = 0.3
    ε_logit :: Float64 = 0.5            # logit (Gumbel) scale on the sector choice
    N_w   :: Int       = 200
    w_min :: Float64   = 0.0
    w_max :: Float64   = 40.0
end

Base.Broadcast.broadcastable(p::SectoralParams) = Ref(p)

const params = SectoralParams()


# Utility: CRRA felicity `u_crra` is provided by HouseholdStages.


# Switching-cost matrix (plain economic primitive, not a stage) #
#---------------------------------------------------------------#

"""
The `n_sector × n_sector` switching-cost matrix: zero diagonal (free to
stay), constant off-diagonal penalty `κ` (utils) for any move. Plain data
handed to `SectorSwitchingStage`; not household-stage logic.
"""
switching_cost_matrix(p = params) =
    [i == j ? 0.0 : p.κ for i in 1:length(p.sectors), j in 1:length(p.sectors)]


# Household chain assembly #
#--------------------------#

"""
Build the Roy occupational-choice household block
`ProductivityShock ∘ SectorSwitch ∘ Earnings ∘ ConsumptionSavings`, with
aggregate wealth and per-sector population/wealth moments attached at the end.
"""
function sectoral_household(p = params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :prod   => Discrete(p.prod_grid),
        :sector => Discrete(p.sectors),
    )

    shock  = MarkovStage(layout; axis = :prod, transition_matrix = p.P_prod)
    switch = SectorSwitchingStage(layout;                          # defaults: (; axis = :sector)
        switching_cost = switching_cost_matrix(p),
        ε             = p.ε_logit,
    )
    earnings = WealthChangeStage(layout;                           # defaults: (; axis = :wealth)
        wealth_post = function (; sector, wealth, prod, env)
            s_idx = findfirst(==(sector), p.sectors)
            return (1 + p.r) * wealth + env.w_sector[s_idx] * prod
        end,
    )
    savings = ConsumptionSavingsStage(layout;
        β               = p.β,
        utility         = (cell, c) -> u_crra(c, Val(p.σ)),
        # defaults: (; axis = :wealth, skip_monotonicity_check = false, utility_axes = nothing)
    )

    hh = shock ∘ switch ∘ earnings ∘ savings

    # Per-sector population and wealth shares, plus aggregate wealth.
    sector_pop(s)    = (; sector)         -> sector == s ? 1.0    : 0.0
    sector_wealth(s) = (; sector, wealth) -> sector == s ? wealth : 0.0
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
        pop_ag     = at_end(integrand = sector_pop(:ag),     reduce = sum),
        pop_mfg    = at_end(integrand = sector_pop(:mfg),    reduce = sum),
        pop_svc    = at_end(integrand = sector_pop(:svc),    reduce = sum),
        K_ag       = at_end(integrand = sector_wealth(:ag),  reduce = sum),
        K_mfg      = at_end(integrand = sector_wealth(:mfg), reduce = sum),
        K_svc      = at_end(integrand = sector_wealth(:svc), reduce = sum),
    )
end


# Env builder (plain function) #
#------------------------------#

"The env consumed by the chain: just the sector wage vector (prices fixed)."
sectoral_env(p = params) = (; w_sector = p.w_sector)
