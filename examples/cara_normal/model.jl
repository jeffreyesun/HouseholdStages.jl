####################################################################
# CARA–normal portfolio benchmark — the textbook check on θ*        #
####################################################################

# The textbook CHECK on `MeanVarianceStage`: the classic mean-variance portfolio
# rule. For a small risk and a small premium the interior optimal risky share is
#
#     θ* ≈ (μ − R_f) / (γ · σ²)
#
# with μ = E[R̃] the mean gross risky return, σ² = Var(R̃), γ the risk-aversion
# coefficient. "CARA–normal" is the textbook packaging (CARA utility + normal
# returns gives this rule exactly); here the felicity is the package's CRRA
# `u_crra(c, Val(γ))`, for which the same rule is the Samuelson–Merton MYOPIC
# share — and for CRRA that share is, in the one-period problem, EXACTLY
# wealth-INDEPENDENT: factoring `W^{1-γ}` out of `E[(W·(R_f + θ·excess))^{1-γ}]`
# leaves a θ-objective with no `W` in it. So θ*(x) is flat across the wealth
# grid, and the only gaps to the formula are (i) the `shares` grid step and
# (ii) the local CRRA≈CARA / higher-moment error (the formula is a 2nd-order
# mean-variance approximation; the risky leg here is a two-point gamble, not a
# normal).
#
# The household "block" is a SINGLE existing library stage — no bespoke stage —
#
#     Portfolio = MeanVarianceStage(:wealth, fine share grid)
#
# The benchmark is the ONE-PERIOD mean-variance problem: with a terminal value
# `V_end(w) = u_crra(w)`, one `backward!` of `MeanVarianceStage` solves
# `max_θ E[u(b'·(R_f + θ·(R_k − R_f)))]` per wealth cell and seats `θ*(x)`. This
# isolates the static portfolio rule from the multi-period human-wealth /
# Merton hedging effects that a full consumption-savings steady state would
# layer on (those tilt the young toward equity — the Cocco–Gomes channel — and
# are a different object from the textbook static share). The streaming
# `(max, +)` of `MeanVarianceStage` should recover θ* up to the two gaps above.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CARANormalParams
    γ :: Float64       = 2.0                        # CRRA / risk aversion in θ* = (μ−R_f)/(γσ²)
    R_f     :: Float64 = 1.02                        # gross risk-free return
    # Two-point risky return, symmetric ±d around the mean μ = R_f + premium, so
    # Var(R̃) = d² EXACTLY. Small premium and small spread ⇒ the local
    # CRRA≈CARA mean-variance approximation is decent and θ* is interior.
    premium :: Float64 = 0.01                        # μ − R_f
    d       :: Float64 = 0.10                        # half-spread; σ² = d² = 0.01
    p_up    :: Float64 = 0.5
    # FINE share grid (step 0.005), extended past 1 so interiority is unambiguous.
    shares  :: Vector{Float64} = collect(0.0:0.005:1.5)
    N_w   :: Int       = 160
    w_min :: Float64   = 0.5
    w_max :: Float64   = 50.0
end

Base.Broadcast.broadcastable(p::CARANormalParams) = Ref(p)

const cara_params = CARANormalParams()

"The mean gross risky return μ = R_f + premium and its two-point realization (μ ± d)."
risky_returns(p::CARANormalParams) = (p.R_f + p.premium) .+ (p.d .* [-1.0, 1.0])

"""
The textbook mean-variance share `θ* = (μ − R_f) / (γ · σ²)` for this calibration
(`μ = R_f + premium`, `σ² = d²`) — the closed-form target the seated policy is
checked against.
"""
textbook_share(p::CARANormalParams = cara_params) = p.premium / (p.γ * p.d^2)


# Household block — a single MeanVarianceStage #
#----------------------------------------------#

"The single-axis wealth layout the benchmark lives on."
cara_layout(p = cara_params) =
    GriddedLayout(:wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log))

"""
The CARA–normal benchmark block: a single `MeanVarianceStage` on `:wealth` with
the FINE share grid over which the seated `θ*(x)` is read. One existing stage,
no bespoke household stage.
"""
cara_portfolio(layout = cara_layout(); p = cara_params) =
    MeanVarianceStage(layout; shares = p.shares, risk_free = p.R_f,
                      risky_returns = risky_returns(p), probs = [1 - p.p_up, p.p_up])

"""
The one-period terminal value `V_end(w) = u_crra(w, Val(γ))` evaluated on the
wealth grid — the continuation against which the static portfolio share is
optimized.
"""
cara_terminal_value(layout = cara_layout(); p = cara_params) =
    u_crra.(collect(axis_grid(layout, :wealth)), Val(p.γ))
