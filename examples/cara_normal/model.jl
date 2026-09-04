####################################################################
# CARA–normal portfolio benchmark — the textbook check on θ*        #
####################################################################

# The textbook CHECK on `GaussianLoadingStage` in its canonical portfolio reading
# (anchor = R_f, increment = the Gaussian excess return): the classic mean-variance
# portfolio rule. For a small risk and a small premium the interior optimal risky share is
#
#     θ* ≈ (μ − R_f) / (γ · σ²)
#
# with μ = E[R̃] the mean gross risky return, σ² = Var(R̃), γ the risk-aversion
# coefficient. "CARA–normal" is the textbook packaging (CARA utility + normal
# returns gives this rule exactly) — and the stage's return contract IS that
# setup natively: the excess return is a (±8σ-truncated) Gaussian and θ is
# CONTINUOUS, so the benchmark's distributional assumption holds by
# construction rather than by approximation. The felicity is
# the package's CRRA `u_crra(c, Val(γ))`, for which the same rule is the
# Samuelson–Merton MYOPIC share — and for CRRA that share is, in the one-period
# problem, EXACTLY wealth-INDEPENDENT: factoring `W^{1-γ}` out of
# `E[(W·(R_f + θ·excess))^{1-γ}]` leaves a θ-objective with no `W` in it. So
# θ*(x) is flat across the wealth grid, and the gaps to the formula are only
# (i) the log-grid interpolation of the continuation at off-node landings and
# (ii) the local CRRA≈CARA / higher-moment error (the formula is a 2nd-order
# mean-variance approximation of the CRRA objective; the ±8σ truncation is
# immaterial at these scales). θ* itself is solved continuously (scan +
# Newton), so the choice contributes no discretization error of its own.
#
# The household "block" is a SINGLE existing library stage — no bespoke stage —
#
#     Portfolio = GaussianLoadingStage(:wealth; loading_bounds = (0.0, 1.5))
#
# The benchmark is the ONE-PERIOD mean-variance problem: with a terminal value
# `V_end(w) = u_crra(w)`, one `backward!` of `GaussianLoadingStage` solves
# `max_θ E[u(b'·(R_f + θ·(μ_x + σ_x·Z)))]` per wealth cell and seats `θ*(x)`.
# This isolates the static portfolio rule from the multi-period human-wealth /
# Merton hedging effects that a full consumption-savings steady state would
# layer on (those tilt the young toward equity — the Cocco–Gomes channel — and
# are a different object from the textbook static share). The continuous
# solver of `GaussianLoadingStage` should recover θ* up to the two gaps above.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CARANormalParams
    γ :: Float64       = 2.0                        # CRRA / risk aversion in θ* = (μ−R_f)/(γσ²)
    R_f     :: Float64 = 1.02                        # gross risk-free return
    # Gaussian excess return with mean `premium` and sd matched to a two-point
    # gamble ±d (σ_x = d at p_up = ½), so Var = d² EXACTLY. Small
    # premium and small spread ⇒ the local CRRA≈CARA mean-variance approximation
    # is decent and θ* is interior.
    premium :: Float64 = 0.01                        # μ − R_f
    d       :: Float64 = 0.10                        # half-spread; σ² = d² = 0.01
    p_up    :: Float64 = 0.5
    # Share interval extended past 1 so interiority is unambiguous (θ continuous).
    share_bounds :: Tuple{Float64, Float64} = (0.0, 1.5)
    N_w   :: Int       = 160
    w_min :: Float64   = 0.5
    w_max :: Float64   = 50.0
end

Base.Broadcast.broadcastable(p::CARANormalParams) = Ref(p)

const cara_params = CARANormalParams()

"""
The Gaussian excess-return moments `(μ_x, σ_x)`: the two-point gamble
`premium ± d` (up w.p. `p_up`) moment-matched — `μ_x = premium + (2p_up − 1)·d`,
`σ_x = 2d·√(p_up(1 − p_up))`, which collapse to `(premium, d)` at `p_up = ½`.
"""
excess_moments(p::CARANormalParams = cara_params) =
    (p.premium + (2p.p_up - 1) * p.d, 2p.d * sqrt(p.p_up * (1 - p.p_up)))

"""
The textbook mean-variance share `θ* = (μ − R_f) / (γ · σ²)` for this calibration
(`μ − R_f = μ_x`, `σ² = σ_x²`) — the closed-form target the seated policy is
checked against.
"""
textbook_share(p::CARANormalParams = cara_params) =
    (m = excess_moments(p); m[1] / (p.γ * m[2]^2))


# Household block — a single GaussianLoadingStage #
#----------------------------------------------#

"The single-axis wealth layout the benchmark lives on."
cara_layout(p = cara_params) =
    GriddedLayout(:wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log))

"""
The CARA–normal benchmark block: a single `GaussianLoadingStage` on `:wealth` with
the CONTINUOUS share interval `share_bounds` over which the seated `θ*(x)` is
solved. One existing stage, no bespoke household stage.
"""
function cara_portfolio(layout = cara_layout(); p = cara_params)
    μx, σx = excess_moments(p)
    return GaussianLoadingStage(layout; anchor = p.R_f, increment_mean = μx,
                             increment_sd = σx, loading_bounds = p.share_bounds)
end

"""
The one-period terminal value `V_end(w) = u_crra(w, Val(γ))` evaluated on the
wealth grid — the continuation against which the static portfolio share is
optimized.
"""
cara_terminal_value(layout = cara_layout(); p = cara_params) =
    u_crra.(collect(axis_grid(layout, :wealth)), Val(p.γ))
