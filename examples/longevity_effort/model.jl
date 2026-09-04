####################################################################
# Longevity effort (Grossman survival / Pijoan-Mas–Ríos-Rull 2014)  #
####################################################################

# A life-cycle household that pays a CONVEX EFFORT COST to RAISE its own
# per-period survival probability — the "value of life" / health-effort margin
# of Grossman (1972) and Pijoan-Mas & Ríos-Rull (2014). The whole point of this
# Part-3 example: survival itself is the retention choice, and the within-period
# problem is THREE existing library stages, in time order, with **no bespoke
# household stage rolled here** —
#
#     Survival ∘ Receipt ∘ ConsumptionSavings
#
# `Survival`  — `RetentionStage` on the `:alive` axis. `K_A = I` (stay on the
#               current alive/dead state) and `K_B = death_kernel` is CERTAIN
#               death (alive → dead w.p. 1, dead → dead absorbing). The blended
#               alive row is `θ·[1,0] + (1−θ)·[0,1] = [θ, 1−θ]`, so the chosen
#               survival probability IS `θ`, bought at convex effort cost
#               `c(θ) = θ²/(2κ)` (in UTILS). The closed form is
#               `V = (death value) + c*(alive value − death value)` — survival
#               effort rises with the continuation value of being alive (the
#               value-of-life mechanism). This is the same `MixingStage` machinery
#               as `examples/insurance`, but the retention axis is `:alive`
#               (survival) rather than `:wealth` (the asset stock), and distinct
#               from `examples/health` where survival is a CapitalInvestment +
#               health-dependent Markov rather than a direct retention choice.
# `Receipt`   — `WealthChangeStage` `a ↦ (1+r)·a + y`: return on assets plus
#               labour income (cash-on-hand).
# `ConsumptionSavings` — `ConsumptionSavingsStage` picks next-period wealth on
#               the grid; `c = x − a'`, CRRA flow plus a per-period flow value of
#               being alive `b̄` (the value-of-life normalization — see below).
#               The dead earn ZERO flow, so the dead-state value stays pinned at
#               0 and `alive value − death value = V_alive > 0` drives `θ`.
#
# IMPORTANT FIDELITY CAVEAT. `RetentionStage`'s effort cost `c(θ)` is a UTILS
# (effort-disutility) cost, entering `V` additively through the Fenchel conjugate
# — it is NOT a resource cost competing with consumption out of the budget. So
# wealth affects survival ONLY through the continuation value of being alive
# (richer ⇒ higher `V_alive` ⇒ more survival effort), not through a medical-
# spending budget drain. The resource-cost version (medical expenditure paid out
# of cash-on-hand) is NOT expressible by `RetentionStage`; it needs a
# `CapitalInvestmentStage` on a health/survival stock (cf. `examples/health`).
# This example is the utils-cost reading honestly, and the catalog's
# "pay convex cost not to transition (on the alive axis)" row.
#
# Why finite-horizon (life-cycle), not stationary. Mortality leaks mass to the
# absorbing dead state with no birth source, so there is no nondegenerate
# stationary ALIVE-mass (catalog gap G2: `Λ' = K·Λ + M·g`). The finite-horizon
# cohort sidesteps this honestly: a cohort born alive at `w0` decays along the
# chosen survival curve over the life cycle. Total grid mass is conserved (it
# accumulates in `dead`). This mirrors `examples/health/steady_state.jl` exactly.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct LongevityParams
    β          :: Float64 = 0.96       # patience
    σ          :: Float64 = 2.0        # CRRA
    r          :: Float64 = 0.03       # exogenous net return on wealth
    y          :: Float64 = 1.0        # labour income while alive
    flow_alive :: Float64 = 11.0       # per-period flow VALUE OF BEING ALIVE (value-of-life normalization)
    cost_curvature :: Float64 = 0.05   # κ in the effort cost c(θ)=θ²/(2κ) (UTILS); larger ⇒ cheaper survival
    N_age      :: Int     = 40         # life-cycle length (ages)
    w0         :: Float64 = 5.0        # wealth at birth (cohort starts here, alive)
    N_w        :: Int     = 80         # wealth grid points
    w_min      :: Float64 = 0.0
    w_max      :: Float64 = 40.0
end

Base.Broadcast.broadcastable(p::LongevityParams) = Ref(p)

const longevity_params = LongevityParams()


# Death kernel (plain economic data — a row-stochastic :alive transition) #
#-------------------------------------------------------------------------#

"""
The CERTAIN-death kernel `K_B[from, to]` on the 2-state `:alive` axis
`[:alive, :dead]`: an alive agent transitions to dead w.p. 1 (`[0, 1]`) and the
dead state is absorbing (`[0, 1]`). Handed to `RetentionStage` as its `exit_kernel`
(`K_A = I` is supplied by `RetentionStage`); the blended alive row is then
`[θ, 1−θ]`, so the retention weight `θ` IS the survival probability. Plain data,
not a household stage.
"""
death_kernel() = [0.0 1.0;
                  0.0 1.0]


# Household chain assembly — THREE library stages, NO bespoke stage #
#------------------------------------------------------------------#

"""
Build the longevity-effort household block `Survival ∘ Receipt ∘ ConsumptionSavings`
with `living_mass = ∫ 1{alive} dΛ` and `wealth_living = ∫ wealth·1{alive} dΛ`
moments attached. Three existing stages, no bespoke household stage: survival is a
`RetentionStage` on the `:alive` axis (`K_A = I`, `K_B = certain death`), so the
retention weight `θ` is the survival probability bought at convex utils effort
cost. The dead earn zero flow utility (`cell.alive == :dead`), pinning the
death-state value at 0 so `θ` is driven by the value of being alive.
"""
function longevity_household(p = longevity_params)
    wealth_axis = GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log)
    layout = GriddedLayout(
        :wealth => wealth_axis,
        :alive  => Discrete([:alive, :dead]),
    )

    survival = RetentionStage(layout; axis = :alive,
        exit_kernel    = death_kernel(),                 # K_B: certain death; K_A = I (survive) built in
        cost_curvature = p.cost_curvature)
    receipt = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; wealth, env) -> (1 + env.r) * wealth + env.y)
    savings = ConsumptionSavingsStage(layout;
        β            = p.β,
        utility      = (cell, c) -> cell.alive == :dead ? 0.0 :
                                         p.flow_alive + u_crra(c, Val(p.σ)),
        utility_axes = (:alive,))    # utility reads :alive beyond the operative :wealth axis

    hh = survival ∘ receipt ∘ savings
    return define_moments!(hh;
        living_mass   = at_end(integrand = (; alive)         -> alive == :alive ? 1.0 : 0.0,    reduce = sum),
        wealth_living = at_end(integrand = (; alive, wealth) -> alive == :alive ? wealth : 0.0, reduce = sum))
end


# Exogenous prices (plain function, partial equilibrium) #
#--------------------------------------------------------#

"The exogenous env for the longevity household: net return `r` and labour income `y`."
longevity_env(p = longevity_params) = (; r = p.r, y = p.y)
