###############################################################
# Sovereign / consumer default — incomplete-markets w/ default #
###############################################################

# A borrower that each period chooses REPAY vs DEFAULT on its outstanding
# debt (Eaton–Gersovitz 1981; Arellano 2008; Chatterjee–Corbae–Nakajima–
# Ríos-Rull 2007). The point of this example: the entire within-period
# problem is FIVE existing library stages, in time order, with **no bespoke
# household stage rolled here** —
#
#     IncomeShock ∘ DefaultChoice ∘ DebtReset ∘ Receipt ∘ Savings ∘ Readmission
#
# `IncomeShock`  — `MarkovStage` on the income axis (endowment process).
# `DefaultChoice`— `DefaultStage`: a gated argmax on a 2-level `:status`
#                  axis (1 = good standing, 2 = excluded/in default). A
#                  good-standing agent chooses repay (stay 1, score 0) or
#                  default (→2, score `−χ`); an EXCLUDED agent is gated by
#                  `avail` to remain excluded (it cannot re-enter markets at
#                  will). The continuation `V_end` at each status carries
#                  that branch's value — wired by the stages composed AFTER,
#                  exactly as the framework intends.
# `DebtReset`    — `WealthChangeStage` reading `cell.status`: an excluded /
#                  defaulting agent carries zero assets (debt discharged); a
#                  good-standing repayer keeps `a`. The state consequence of
#                  defaulting is a *following* stage, not baked into the
#                  choice (mirrors BuyHomeStage ∘ WealthChangeStage).
# `Receipt`      — `WealthChangeStage` reading `cell.status`: cash-on-hand
#                  `x = (1+r)·a + earnings`. An excluded agent suffers the
#                  income haircut `(1−λ)·y` (the Arellano output cost) and
#                  has no debt; a good-standing repayer earns `y` and repays
#                  its debt (`a < 0`) out of `x`.
# `Savings`      — `ConsumptionSavingsStage` picks next-period assets
#                  `a' ∈ grid` (the grid floor `a_min < 0` is the borrowing
#                  limit); `c = x − a'` at the exogenous risk-free unit bond
#                  price `q = 1`.
# `Readmission`  — `MarkovStage` on the `:status` axis: an excluded agent
#                  regains good standing next period w.p. `ψ` (a geometric
#                  exclusion spell, mean `1/ψ` periods); a good-standing
#                  agent stays good. This PERSISTENT cost — not the one-shot
#                  penalty — is what deters always-default and delivers a
#                  non-degenerate stationary default rate driven by income
#                  risk, exactly as Eaton–Gersovitz / Arellano intend.
#
# Persistence is the whole mechanism: without an exclusion spell, default is
# a costless per-period arbitrage (discharge debt, re-borrow, repeat) and
# the default rate degenerates to 1. The `Readmission` MarkovStage is the
# canonical fix and is itself a library stage. Prices are exogenous (a
# risk-free unit bond, partial equilibrium): no market to clear, so the
# "outer loop" is a single `solve_steady_state_given_env!`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct DefaultParams
    β :: Float64       = 0.92                      # impatience drives the debt motive
    σ :: Float64       = 2.0                       # CRRA
    r :: Float64       = 0.02                      # exogenous risk-free rate (q = 1/(1+r) folded into receipt)
    λ :: Float64       = 0.40                      # default income haircut (Arellano output cost)
    χ :: Float64       = 0.02                      # extra flat per-period utility cost of exclusion
    ψ :: Float64       = 0.15                      # readmission prob (mean exclusion spell 1/ψ ≈ 6.7 periods)
    y_grid :: Vector{Float64} = [0.7, 1.0, 1.3]
    P_y    :: Matrix{Float64} = [0.6 0.3 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.3 0.6]
    N_a   :: Int       = 200
    a_min :: Float64   = -0.35                     # borrowing limit (debt is a < 0)
    a_max :: Float64   = 4.0
end

Base.Broadcast.broadcastable(p::DefaultParams) = Ref(p)

const default_params = DefaultParams()


# Household chain assembly #
#--------------------------#

"""
Build the default household block
`IncomeShock ∘ DefaultChoice ∘ DebtReset ∘ Receipt ∘ Savings ∘ Readmission`, with
`mean_assets = ∫ a dΛ` and `excluded_rate = ∫ 1{status = excluded} dΛ` attached. The repay/default
branches are wired purely by composition and `cell.status`-reading wealth closures.
"""
function default_household(p = default_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :linear),
        :income => Discrete(p.y_grid),
        :status => Discrete([1, 2]),   # 1 = good standing, 2 = excluded / in default
    )

    shock = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)

    # DefaultChoice. The flat −χ is only the per-period flow cost of exclusion; the heavier
    # punishment (the income haircut, the discharged debt, the multi-period spell) is carried by
    # the following branches and the readmission stage.
    excl_gate(before, after) = before == 1 ? true : after == 2
    default = DefaultStage(layout; default_penalty = p.χ, avail = excl_gate) # defaults: (; axis = :status, default_index = 2)

    # DebtReset: defaulting discharges the debt.
    reset = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; status, wealth) -> status == 2 ? 0.0 : wealth)

    # Receipt: cash-on-hand. An excluded agent takes the Arellano output cost `(1−λ)·y` and has no
    # debt to service; a good-standing repayer earns `y` and services `a < 0` out of it.
    receipt = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; status, income, wealth, env) ->
            status == 2 ? (1 - env.λ) * income
                        : (1 + env.r) * wealth + income)

    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth) # defaults: (; utility_axes = nothing, skip_monotonicity_check = false)

    # Readmission: an excluded agent regains good standing next period w.p. ψ — the geometric
    # exclusion spell that is the persistent cost of default. `T[from, to]` is row-stochastic.
    readmit = MarkovStage(layout; axis = :status,
        transition_matrix = [1.0 0.0; p.ψ (1 - p.ψ)])

    hh = shock ∘ default ∘ reset ∘ receipt ∘ savings ∘ readmit
    return define_moments!(hh;
        mean_assets   = at_end(integrand = :wealth, reduce = sum),
        excluded_rate = at_end(integrand = (; status) -> status == 2 ? 1.0 : 0.0, reduce = sum),
    )
end


# Exogenous prices (plain functions, no AbstractBlock) #
#------------------------------------------------------#

"The exogenous environment: the risk-free rate and the default income haircut."
default_env(p = default_params) = (; p.r, p.λ)
