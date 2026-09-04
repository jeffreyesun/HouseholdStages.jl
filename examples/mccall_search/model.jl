######################################################################
# McCall (1970) reservation-wage search — household block            #
######################################################################

# The classic McCall sequential-search model, expressed ENTIRELY as a `∘` chain of
# existing HouseholdStages library stages — no bespoke household stage. The state is
# `(:wage, :emp)`: an unemployed worker holds a wage OFFER drawn from a distribution
# `F` over the wage grid and decides whether to ACCEPT it (become employed at that
# wage) or REJECT it (stay unemployed and redraw next period). The employed separate
# exogenously at rate `δ`. The accept margin is a RESERVATION WAGE: accept iff the
# offer `w` clears `w ≥ w*`.
#
# The within-period problem decomposes into five stages, in time order (left → right):
#
#     AcceptReject ∘ FlowUtility ∘ Discount ∘ Separation ∘ OfferDraw
#
# - `AcceptReject` (`ArgmaxStage` on the 2-level `:emp` axis) — the (max,+) accept/
#   reject choice. The reward `M[after, before]` GATES infeasible moves with `-Inf`:
#   from an unemployed origin BOTH destinations are open (accept → employed, or stay
#   unemployed); from an employed origin only "stay employed" is open (`-Inf` on
#   employed → unemployed — McCall workers cannot voluntarily quit; separation is
#   exogenous and handled downstream). The `:wage` axis is a SPECTATOR, so the
#   continuation `V_end[emp]` rises with the held wage while `V_end[unemp]` does not
#   (the unemployed redraw), producing the reservation-wage cutoff.
# - `FlowUtility` (`UtilityStage`) — period flow `u(w)` if employed, `u(b_u)` if
#   unemployed (the benefit `b_u` rides `env`).
# - `Discount` (`TimeDiscountingStage`) — `V_start = β·V_end`.
# - `Separation` (`MarkovStage` on `:emp`) — employed → unemployed w.p. `δ`,
#   unemployed stay unemployed (an exogenous, end-of-period job-destruction shock).
# - `OfferDraw` (`MarkovStage` on `:wage`, `:emp`-dependent) — an unemployed worker's
#   next wage is an i.i.d. draw from the offer distribution `F` (every row of the
#   kernel equals `F`); an employed worker keeps their wage (identity). A worker who
#   just separated is unemployed by this point and therefore redraws.
#
# Period value: V_unemp(w) = max{ u(b_u) + βE V', u(w) + βE V'_emp }, the inner
# expectations resolving the separation + offer-draw transitions; the accept branch
# wins iff w ≥ w*. References: McCall (1970, QJE); Ljungqvist & Sargent, ch. 6.
#
# Library stages used (NO bespoke household stage): ArgmaxStage, UtilityStage,
# TimeDiscountingStage, MarkovStage (×2).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct McCallParams
    β   :: Float64 = 0.96          # discount factor
    σ   :: Float64 = 2.0           # CRRA curvature
    b_u :: Float64 = 0.70          # flow income while unemployed (benefit)
    δ   :: Float64 = 0.10          # exogenous job-separation rate

    # Wage offer grid + log-normal offer distribution F.
    N_w     :: Int     = 60        # number of wage points (a fine, "continuous" wage axis)
    w_lo    :: Float64 = 0.20
    w_hi    :: Float64 = 2.50
    offer_μ :: Float64 = 0.0       # log-normal location (median offer = exp(μ) = 1.0)
    offer_s :: Float64 = 0.40      # log-normal log-scale
end

Base.Broadcast.broadcastable(p::McCallParams) = Ref(p)

const mccall_params = McCallParams()


# Wage grid + offer-distribution helpers (plain economic data) #
#--------------------------------------------------------------#

"""
The wage offer grid: `N_w` evenly spaced points on `[w_lo, w_hi]`. A plain grid of
labels — both the spectator state and the support of the offer distribution `F`.
"""
wage_grid(p = mccall_params) = collect(range(p.w_lo, p.w_hi; length = p.N_w))

"""
Discretized offer distribution `F` over the wage grid: a log-normal density (median
`exp(offer_μ)`, log-scale `offer_s`) sampled at the grid points and normalized to a
pmf. The unemployed draw a fresh offer from `F` each period.
"""
function offer_distribution(grid::AbstractVector, p = mccall_params)
    dens = @. exp(-((log(grid) - p.offer_μ)^2) / (2 * p.offer_s^2)) / grid
    return dens ./ sum(dens)
end

"""
Row-stochastic offer-draw kernel whose every row equals the offer pmf `F`: an
unemployed worker's next wage is an i.i.d. draw from `F`, independent of the offer
currently held. (Fed to `MarkovStage` as the unemployed branch of `OfferDraw`.)
"""
offer_kernel(F::AbstractVector) = repeat(reshape(F, 1, :), length(F), 1)

"""
The `n×n` identity kernel — an employed worker keeps their current wage (the employed
branch of `OfferDraw`).
"""
identity_kernel(n::Int) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached McCall search block
`AcceptReject ∘ FlowUtility ∘ Discount ∘ Separation ∘ OfferDraw`, with the employment
rate and the (employed) wage mass attached as moments. The `:emp` axis is 1 = unemployed,
2 = employed; the benefit `b_u` is read from `env`. Five existing library stages, no
bespoke household stage: the accept/reject choice is a gated `ArgmaxStage` whose `-Inf`
entries forbid voluntary quits, and the `:wage` spectator delivers the reservation-wage
cutoff.
"""
function mccall_household(p = mccall_params)
    grid = wage_grid(p)
    n    = length(grid)
    layout = GriddedLayout(
        :wage => Discrete(grid),
        :emp  => Discrete([:unemp, :emp]),   # 1 = unemployed, 2 = employed
    )

    # Accept/reject reward M[after, before] on :emp (rows = destination, cols = origin).
    #   origin unemp (col 1): stay unemp (M[1,1]=0) OR accept→emp (M[2,1]=0)   — both open
    #   origin emp   (col 2): cannot quit (M[1,2]=-Inf); stay employed (M[2,2]=0)
    accept_reward = [0.0   -Inf;
                     0.0    0.0]

    # Separation T[from, to] on :emp: employed → unemp w.p. δ; unemployed stay unemp.
    sep_T = [1.0      0.0;
             p.δ      1 - p.δ]

    F   = offer_distribution(grid, p)
    F_K = offer_kernel(F)        # unemployed: redraw an offer from F
    I_K = identity_kernel(n)     # employed: keep the wage

    accept_reject = ArgmaxStage(layout; axis = :emp, reward = accept_reward)
    flow_utility  = UtilityStage(layout;
        utility = (; emp, wage, env) -> emp == :emp ? u_crra(wage, Val(p.σ)) :
                                                      u_crra(env.b_u, Val(p.σ)))
    discount   = TimeDiscountingStage(layout; β = p.β)
    separation = MarkovStage(layout; axis = :emp, transition_matrix = sep_T)
    offer_draw = MarkovStage(layout; axis = :wage,
        transition_matrix = (; emp) -> emp == :unemp ? F_K : I_K)

    hh = accept_reject ∘ flow_utility ∘ discount ∘ separation ∘ offer_draw
    return define_moments!(hh;
        employment = at_end(integrand = (; emp) -> emp == :emp ? 1.0 : 0.0, reduce = sum),
        wage_emp   = at_end(integrand = (; wage, emp) -> emp == :emp ? wage : 0.0, reduce = sum),
    )
end

"""
Env for the McCall partial-equilibrium experiment: the unemployment benefit `b_u`
consumed by the flow-utility closure.
"""
mccall_env(p = mccall_params) = (; b_u = p.b_u)
