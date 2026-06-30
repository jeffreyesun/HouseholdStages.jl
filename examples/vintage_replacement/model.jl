###############################################################
# Vintage technology adoption / replacement — regenerative      #
# optimal stopping (Rust 1987; Cooper–Haltiwanger replacement)  #
###############################################################

# A household operates a productive durable / technology of VINTAGE `v` whose
# quality decays stochastically each period (obsolescence). Each period the
# agent chooses KEEP (let the vintage drift down via the obsolescence Markov)
# or ADOPT (pay a fixed cost `F` and reset the vintage to the newest level
# `v_top`). Adopting is worthwhile only once the gain — higher output net of the
# fixed cost — turns positive, i.e. once the vintage has decayed far enough.
# This is the §5(i) **regenerative stopping** problem: the state regenerates to a
# fixed point (the top vintage) at a cost, the classic Rust (1987) bus-engine
# replacement / Cooper–Haltiwanger capital-replacement structure, here embedded
# in an income-fluctuations consumption-savings household.
#
# The point of the example, as with every §5 model: the entire within-period
# block is a `∘`-composition of EXISTING library stages — no bespoke stage.
#
# THE COUPLING. The adopt choice must move TWO axes at once: it resets the
# `:vintage` axis to the top AND charges a resource cost `F` on the `:wealth`
# axis — and the cost must be charged ONLY to a fresh adopter, never to a
# continuing top-vintage keeper sitting on the identical post-choice cell. A
# plain gated `ArgmaxStage` on `:vintage` cannot express this (a fresh adopter
# and a standing top-vintage owner land on the same `vintage = v_top` cell, so a
# following `WealthChangeStage` charges them identically — the exact
# distinguishability wall documented in `durable_housing`'s one-time-price note).
# It IS expressible via the **auxiliary-choice-axis pattern** (Route A, as in
# `examples/habit`): route the keep/adopt decision through a separate
# `:adopt_choice` axis, so downstream stages read the *decision* — not just the
# post-choice vintage — and can reset the vintage AND charge the cost to
# adopters only, with the pre-choice vintage still live.
#
# Household block (time order), all existing library stages, NO bespoke stage:
#
#   IncomeShock ∘ Choose ∘ SetVintage ∘ Receipt ∘ PayCost ∘ Forget ∘ Savings ∘ Depreciate
#
# `IncomeShock` — `MarkovStage(:income)`: 2-state labour-income risk.
# `Choose`      — `ArgmaxStage(:adopt_choice)` grows the choice axis 1→2 ({keep, adopt}),
#                 reward 0: it just compares the two continuations `V_end[keep]`, `V_end[adopt]`.
# `SetVintage`  — `WealthChangeStage(:vintage)`: adopters (`adopt_choice == 2`) reset to `v_top`;
#                 keepers stay at their current vintage (reads BOTH the choice and the old vintage).
# `Receipt`     — `WealthChangeStage(:wealth)`: cash-on-hand `(1+r)·b + w·y + θ·v` on the
#                 POST-adoption vintage `v` (so adopting raises this period's output too).
# `PayCost`     — `WealthChangeStage(:wealth)`: adopters pay the fixed cost `F` out of cash-on-hand
#                 (charged after receipt, so `c > 0` stays feasible; reads the choice → adopters only).
# `Forget`      — `ForgetfulSumStage(:adopt_choice)`: collapse the auxiliary choice axis 2→1.
# `Savings`     — `ConsumptionSavingsStage(:wealth)`: pick `b'`, `c = cash − b'`, CRRA over `c`.
# `Depreciate`  — `MarkovStage(:vintage)`: obsolescence drift — the vintage falls one level
#                 w.p. `π_dep`, carried into next period. This decay is what makes re-adoption recur.
#
# Structurally this is the same kernel-choice sandwich as `habit`
# (`Choose ∘ {axis transforms reading the choice} ∘ ForgetfulSum`), here applied
# to a stopping problem rather than a smooth habit. Partial equilibrium: `r, w`
# and the technology parameters `θ, F` are exogenous — a single inner V/Λ solve.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct VintageReplacementParams
    β :: Float64       = 0.93                     # discount factor
    σ :: Float64       = 2.0                      # CRRA over consumption
    r :: Float64       = 0.03                     # exogenous return on liquid wealth
    w :: Float64       = 1.0                      # wage (labour-income scale)
    θ :: Float64       = 1.0                      # output per unit of vintage quality (θ·v added to cash)
    F :: Float64       = 1.0                      # fixed adoption / replacement cost (resource units)
    v_grid :: Vector{Float64} = [1.0, 1.4, 1.8, 2.2]   # vintage quality levels; index end = newest
    π_dep  :: Float64  = 0.25                     # per-period obsolescence: vintage drops one level
    y_grid :: Vector{Float64} = [0.8, 1.2]
    P_y    :: Matrix{Float64} = [0.85 0.15;
                                 0.15 0.85]
    N_w   :: Int       = 120
    w_min :: Float64   = 0.0
    w_max :: Float64   = 30.0
end

Base.Broadcast.broadcastable(p::VintageReplacementParams) = Ref(p)

const vintage_replacement_params = VintageReplacementParams()


# Obsolescence Markov on the vintage axis #
#-----------------------------------------#

"""
The vintage obsolescence transition `P_v[from, to]` (row-stochastic): each period the
vintage drops one quality level w.p. `π_dep`, else stays; the bottom level is an absorbing
floor (decays no further). This is the drift that erodes the durable and so makes periodic
re-adoption optimal.
"""
function vintage_depreciation_matrix(p = vintage_replacement_params)
    n = length(p.v_grid)
    P = zeros(n, n)
    P[1, 1] = 1.0
    for v in 2:n
        P[v, v - 1] = p.π_dep
        P[v, v]    += 1 - p.π_dep
    end
    return P
end


# Household chain assembly #
#--------------------------#

"""
Build the vintage-replacement block
`IncomeShock ∘ Choose ∘ SetVintage ∘ Receipt ∘ PayCost ∘ Forget ∘ Savings ∘ Depreciate`
via the auxiliary-choice-axis pattern (eight existing stages, no bespoke stage). The keep/adopt
decision rides a transient `:adopt_choice` axis so downstream stages reset the vintage AND charge
the fixed cost to adopters only. Moments attached: `mean_vintage = ∫ v dΛ` and `top_share`
(`∫ 1{v = v_top} dΛ`); the per-period adoption rate is read off the `Choose` policy in the driver.
"""
function vintage_replacement_household(p = vintage_replacement_params)
    v_top = p.v_grid[end]
    axes_base = (:wealth  => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
                 :vintage => GriddedContinuous(p.v_grid),
                 :income  => Discrete(p.y_grid))
    # Pre-/post-choice stages see `:adopt_choice` as a SINGLETON; the choice block sees it FULL
    # ({1 = keep, 2 = adopt}). `Choose` grows 1→2, `Forget` collapses 2→1.
    block = GriddedLayout(axes_base..., :adopt_choice => Discrete([1]))
    full  = GriddedLayout(axes_base..., :adopt_choice => Discrete([1, 2]))

    shock = MarkovStage(block; axis = :income, transition_matrix = p.P_y)
    # Reward 0: the argmax just picks the higher of the two continuations (keep vs adopt).
    choose = ArgmaxStage(full; axis = :adopt_choice, reward = zeros(2, 1))
    setvintage = WealthChangeStage(full; axis = :vintage,                       # adopt → reset to v_top
        wealth_post = (; vintage, adopt_choice) -> adopt_choice == 2 ? v_top : vintage)
    receipt = WealthChangeStage(full; axis = :wealth,                           # cash-on-hand on post-adoption vintage
        wealth_post = (; wealth, income, vintage, env) -> (1 + env.r) * wealth + env.w * income + env.θ * vintage)
    paycost = WealthChangeStage(full; axis = :wealth,                           # adopters pay the fixed cost F
        wealth_post = (; wealth, adopt_choice, env) -> adopt_choice == 2 ? wealth - env.F : wealth)
    forget = ForgetfulSumStage(full; axis = :adopt_choice)
    savings = ConsumptionSavingsStage(block;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),                        # defaults: (; axis = :wealth)
    )
    depreciate = MarkovStage(block; axis = :vintage,
        transition_matrix = vintage_depreciation_matrix(p))

    hh = shock ∘ choose ∘ setvintage ∘ receipt ∘ paycost ∘ forget ∘ savings ∘ depreciate
    return define_moments!(hh;
        mean_vintage = at_end(integrand = :vintage, reduce = sum),
        top_share    = at_end(integrand = (; vintage) -> vintage == v_top ? 1.0 : 0.0,
                              reduce = sum))
end


# Exogenous prices / technology (plain function, partial equilibrium) #
#---------------------------------------------------------------------#

"Exogenous env: return, wage, vintage-output scale, and the fixed adoption cost (no market to clear)."
vintage_replacement_env(p = vintage_replacement_params) = (; r = p.r, w = p.w, θ = p.θ, F = p.F)
