######################################################################
# Entrepreneurship & wealth inequality (Quadrini 2000; Cagetti–De Nardi 2006) #
######################################################################

# Occupational choice — worker vs (productive, risky, limited-liability)
# ENTREPRENEUR — as the engine of wealth concentration (Quadrini 2000, RED,
# "Entrepreneurship, Saving and Social Mobility"; Cagetti & De Nardi 2006, JPE,
# "Entrepreneurship, Frictions, and Wealth"). High-productivity households sort
# into entrepreneurship, capture the productivity boost on their invested
# wealth, and so accumulate into the top tail — a small entrepreneurial elite
# holds a disproportionate share of wealth.
#
# READ THIS FIRST — what is and is not a straight composition.
# ------------------------------------------------------------------
# The catalog status is ◐ (MODEL_CATALOG §2). The LITERAL occupational model —
# a worker spine `Receipt ∘ Savings` and an entrepreneur spine
# `Receipt ∘ Savings ∘ Business(MeanVarianceStage)` with an EXTRA risk stage —
# CANNOT be joined by `⊕`/`product`. `product` requires every leg to have the
# IDENTICAL concrete Spec type (`ProductStageSpec` asserts
# `typeof(s) === first_type`), and `ChainStageSpec` is parameterised on the
# tuple of its component spec types — so a 4-stage chain and a 5-stage chain are
# DIFFERENT concrete types and the assertion fails (verified empirically — see
# README.md §"The give-up"). That is the documented ◐: distinct stage CHAINS per
# leg is per-axis gating the library does not offer.
#
# The catalog names two clean exits, "a wealth-floor or a fully separate `⊕`
# leg." This file takes the **separate-`⊕`-leg** exit: make the two legs
# STRUCTURALLY UNIFORM by giving BOTH the same chain
#
#     Receipt ∘ Savings ∘ Business ∘ Realize ∘ ZShock
#
# and letting the worker carry a DEGENERATE `Business` (a `MeanVarianceStage`
# whose risky returns equal the risk-free return — a safe asset) and the
# identity branch of `Realize`. The two legs then differ ONLY in the VALUES /
# the captured Bool fed to the shared stages — exactly the
# `examples/discount_heterogeneity` device (one builder, parameterised by value,
# defined at one syntactic site so the closure TYPES match and the Spec types
# are identical). `product` accepts them, and an `ArgmaxStage` over the
# `:occupation` axis seats the occupational CHOICE on top. The
# `examples/risk_shifting` Vereshchagina–Hopenhayn **wealth-floor** is the OTHER
# exit (it captures the entrepreneurial risk margin gating-free, no occupation
# axis at all).
#
# What the constant-returns `MeanVarianceStage` can and cannot do here (honest).
# A streaming `MeanVarianceStage` is CONSTANT-returns and its returns cannot
# read the productivity / occupation axis. With a mean PREMIUM the patient rich
# accumulate without bound (no stationary distribution); mean-NEUTRAL, the
# gamble only has option value at the limited-liability floor, so away from the
# floor the occupational value gap is just the constant wage gap and the choice
# is all-or-nothing (knife-edge). The literature's bounding force — DECREASING
# returns to entrepreneurial scale (span of control) — is not expressible in a
# streaming stage. The faithful, library-only resolution used here puts the
# entrepreneur's edge in a TRANSIENT productivity state `z`: the `Realize`
# stage compounds an entrepreneur's invested wealth by `z` (a productive
# entrepreneur's wealth grows fast), but `z` mean-reverts, so the long-run
# multiplier is `< 1` and the distribution is stationary. High-`z` households
# strictly prefer entrepreneurship (to capture the boost) — a genuine
# STATE-DEPENDENT occupational margin — and form the wealth tail. See README.md
# for the full account.
#
# The household block (no bespoke stage anywhere):
#
#     OccChoice ∘ ⊕_occupation{ worker_leg, entrepreneur_leg }
#
# with each leg, in time order,
#
#     Receipt ∘ Savings ∘ Business ∘ Realize ∘ ZShock
#
# `OccChoice`  — `ArgmaxStage` on the 2-level `:occupation` axis: per `(wealth,
#                z)` cell, pick the occupation with the higher continuation,
#                less a fixed cost `κ` of running a business. Free re-choice each
#                period (the reward is independent of the incoming occupation),
#                so occupation is a within-period decision — the genuine
#                Quadrini / Cagetti–De Nardi occupational CHOICE.
# `Receipt`    — `WealthChangeStage` `a ↦ a + wage`: labor income. The worker
#                earns the full wage `w`; the entrepreneur earns a small wage
#                `w_e` (most entrepreneurial income is the business).
# `Savings`    — `ConsumptionSavingsStage` picks the stake `a'` carried into the
#                business / safe asset; `c = x − a'`, CRRA.
# `Business`   — `MeanVarianceStage` on `:wealth`: stake `a'` becomes
#                `a'·(R_f + θ·(R_k − R_f))`, agent picking project intensity θ.
#                The WORKER's `Business` is degenerate (`R_k = R_f`, the safe
#                asset for any θ); the ENTREPRENEUR faces a genuine (mean-neutral)
#                two-point project — pure project RISK, dialled by θ.
# `Realize`    — `WealthChangeStage`: `a ↦ max(z·a, a_floor)` for the
#                ENTREPRENEUR (productivity `z` compounds the invested wealth;
#                limited liability floors a wiped-out project at `a_floor`) and
#                `a ↦ max(a, a_floor)` for the WORKER (the safe asset, floor
#                slack). One shared closure branching on a captured Bool
#                `entrepreneur` — identical closure TYPE both legs.
# `ZShock`     — `MarkovStage` on `:z` (entrepreneurial productivity), placed
#                LAST so `z` transitions for NEXT period after the occupation
#                choice and the business have used THIS period's `z`.
#
# Returns, productivity, and the wage are exogenous (partial equilibrium): no
# market clears, so the outer loop is a single `solve_steady_state_given_env!`.
# `z` mean-reverts to its low state, so the long-run wealth multiplier is `< 1`
# and the distribution is stationary despite transient high-`z` accumulation.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct EntrepreneurshipParams
    β :: Float64       = 0.96                       # discount factor (β·R_f = 0.998: workers hold a modest buffer)
    σ :: Float64       = 2.0                        # CRRA risk aversion
    w   :: Float64     = 0.45                        # worker wage (the safe occupation's labor income)
    w_e :: Float64     = 0.35                        # entrepreneur's (smaller) labor income
    # Entrepreneurial productivity z — a TRANSIENT high state. The Realize stage
    # compounds an entrepreneur's invested wealth by z; z mean-reverts to its low
    # state (P_z heavily favors it), so the long-run multiplier is < 1 (stationary).
    z_grid :: Vector{Float64} = [0.96, 1.08]
    P_z    :: Matrix{Float64} = [0.94 0.06;          # low state is very persistent
                                 0.55 0.45]          # high (productive) state is transient
    # The business technology — a MEAN-NEUTRAL two-point project (E[R_k] = R_f):
    # the entrepreneur's EDGE is the productivity boost z, not a return premium
    # (a constant-returns premium would explode wealth). θ dials pure project risk.
    R_f    :: Float64 = 1.04                          # safe gross return (worker's asset; entrepreneur's risk-free leg)
    R_up   :: Float64 = 1.30                          # business success multiple
    R_dn   :: Float64 = 0.78                          # business failure multiple (E[R_k] = 0.5·1.30 + 0.5·0.78 = 1.04 = R_f)
    p_up   :: Float64 = 0.50                          # success probability
    shares :: Vector{Float64} = collect(range(0.0, 1.0; length = 8))   # project intensity θ
    a_floor :: Float64 = 0.40                         # limited-liability floor on realized wealth
    κ      :: Float64 = 0.05                          # per-period fixed cost of running a business
    N_a   :: Int       = 140
    a_min :: Float64   = 0.0
    a_max :: Float64   = 300.0
end

Base.Broadcast.broadcastable(p::EntrepreneurshipParams) = Ref(p)

const entrepreneurship_params = EntrepreneurshipParams()


# Household chain assembly #
#--------------------------#

"""
Build ONE occupational leg `Receipt ∘ Savings ∘ Business ∘ Realize ∘ ZShock`
against the shared layout (the `:occupation` axis a size-1 singleton, so every
leg has identical input layout and — because the closures are defined HERE, at
one syntactic site, capturing values not differing code — identical Spec TYPE).
The worker and entrepreneur legs differ ONLY in the captured VALUES `wage`,
`business_returns`, and the `entrepreneur` flag — exactly the uniformity
`product` requires. The worker's `Business` is degenerate (`business_returns`
all `= R_f`, a safe asset) and its `Realize` is the floor alone; the
entrepreneur's `Business` is the mean-neutral project and its `Realize` applies
the productivity-`z` compounding under the limited-liability floor.
"""
function entrepreneurship_leg(p::EntrepreneurshipParams; wage::Float64,
                              business_returns::Vector{Float64}, entrepreneur::Bool)
    layout = GriddedLayout(
        :wealth     => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log),
        :z          => Discrete(p.z_grid),
        :occupation => Discrete([1.0]),          # size-1 singleton; product grows it 1 → 2
    )

    receipt = WealthChangeStage(layout; axis = :wealth,                       # labor income
        wealth_post = (; wealth, env) -> wealth + wage)
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)))
    business = MeanVarianceStage(layout; axis = :wealth,                      # safe (worker) or mean-neutral project (entrepreneur)
        shares        = p.shares,
        risk_free     = p.R_f,
        risky_returns = business_returns,
        probs         = [p.p_up, 1 - p.p_up])
    realize = WealthChangeStage(layout; axis = :wealth,                       # entrepreneur: productivity z + limited liability; worker: floor alone
        wealth_post = (; z, wealth, env) ->
            entrepreneur ? max(z * wealth, env.a_floor) : max(wealth, env.a_floor))
    zshock = MarkovStage(layout; axis = :z, transition_matrix = p.P_z)        # z transitions for next period (placed last)

    return receipt ∘ savings ∘ business ∘ realize ∘ zshock
end

"""
The entrepreneurship household block:
`OccChoice ∘ product(worker_leg, entrepreneur_leg; axis = :occupation)`.
Both legs share an identical Spec type (the worker's `Business` is a degenerate
safe-return `MeanVarianceStage` and its `Realize` is the floor alone), so
`product` joins them along `:occupation`; the `OccChoice` `ArgmaxStage` on top
seats the within-period occupational choice (higher continuation, less the fixed
cost `κ` of entrepreneurship). Attaches `mean_wealth = ∫ wealth dΛ` and
`entrepreneur_share = ∫ 1{occupation = 2} dΛ`.
"""
function entrepreneurship_household(p = entrepreneurship_params)
    worker = entrepreneurship_leg(p; wage = p.w,
        business_returns = [p.R_f, p.R_f], entrepreneur = false)             # degenerate: safe asset
    entre  = entrepreneurship_leg(p; wage = p.w_e,
        business_returns = [p.R_up, p.R_dn], entrepreneur = true)            # genuine project risk + productivity boost
    legs = product(worker, entre; axis = :occupation)

    # Free occupational re-choice; reward[after, before] independent of the
    # incoming occupation. Running a business (occupation 2) costs κ.
    occ_layout = legs.buffer.output_layout
    occ_reward = [0.0   0.0;
                  -p.κ  -p.κ]
    occ_choice = ArgmaxStage(occ_layout; axis = :occupation, reward = occ_reward, search = :brute)

    hh = occ_choice ∘ legs
    return define_moments!(hh;
        mean_wealth       = at_end(integrand = :wealth, reduce = sum),
        entrepreneur_share = at_end(integrand = (; occupation) -> occupation > 1.5 ? 1.0 : 0.0,
                                    reduce = sum))
end

"The exogenous entrepreneurship env: the limited-liability floor `a_floor` read by both legs' `Realize`."
entrepreneurship_env(p = entrepreneurship_params; a_floor = p.a_floor) = (; a_floor)
