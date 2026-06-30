####################################################################
# CARA–normal benchmark — seated θ*(x) vs the textbook formula      #
####################################################################

# Not a steady state but a ONE-PERIOD mean-variance check (the file keeps the
# examples' conventional name). With terminal value `V_end(w) = u_crra(w)`, one
# `backward!` of the `MeanVarianceStage` seats the static optimal share `θ*(x)`
# per wealth cell. The check: θ*(x) is interior and (for CRRA) wealth-INDEPENDENT
# — flat across the grid INTERIOR — and close to the closed-form
# `θ* = (μ − R_f)/(γσ²)` up to (i) the `shares` grid step, (ii) the log-grid
# interpolation of the continuation at the off-node landings, and (iii) the
# local CRRA≈CARA / higher-moment error (the formula is a 2nd-order mean-variance
# approximation; the risky leg is a two-point gamble, not a normal).
#
# Grid EDGES are excluded: a landing `w·(R_f ± θ·d)` from a near-edge cell falls
# outside `[w_min, w_max]` and is clamped, breaking the exact homogeneity that
# makes the CRRA share wealth-independent. The check is therefore stated on the
# INTERIOR band — cells whose worst/best-case landing stays on-grid.

include("model.jl")

using Printf, Statistics

"""
Wealth-grid indices whose worst-case (max-share, down) and best-case (max-share,
up) landings both stay within `[w_min, w_max]` — the interior band on which the
homogeneity (hence the wealth-independent CRRA share) is not corrupted by
landing clamping at the grid edges.
"""
function cara_interior(layout, p)
    w   = collect(axis_grid(layout, :wealth))
    smx = maximum(p.shares)
    lo  = p.w_min / (p.R_f - smx * p.d)
    hi  = p.w_max / (p.R_f + smx * p.d)
    return findall(x -> lo <= x <= hi, w)
end

"""
Run the one-period CARA–normal benchmark: build the `MeanVarianceStage`, drive a
single `backward!` from the CRRA terminal value, and compare the seated risky
share `θ*(x)` to the textbook `(μ − R_f)/(γσ²)`. Reports the full-grid range, the
interior-band range/dispersion (the wealth-independence check), the closed-form
target, and the discretization gap. Returns the policy, the interior indices, the
formula value, and the grid.
"""
function cara_benchmark(p = cara_params; verbosity = 1)
    layout = cara_layout(p)
    stage  = cara_portfolio(layout; p)
    V_end  = cara_terminal_value(layout; p)

    backward!(stage, V_end, (;))                       # one static optimization; seats θ*(x)
    θ      = HouseholdStages.policy(stage)             # θ*(x) on the wealth grid
    θ_form = textbook_share(p)
    intr   = cara_interior(layout, p)
    θ_int  = θ[intr]
    Δθ     = p.shares[2] - p.shares[1]

    if verbosity > 0
        μ = p.R_f + p.premium
        @printf "CARA–normal one-period benchmark (γ = %.1f, μ = %.3f, R_f = %.3f, σ² = %.4f)\n" p.γ μ p.R_f p.d^2
        @printf "  textbook θ* = (μ−R_f)/(γσ²) = %.4f\n" θ_form
        @printf "  seated θ*(x), full grid : [%.4f, %.4f]  (edges clamp to the corners)\n" minimum(θ) maximum(θ)
        @printf "  seated θ*(x), interior  : [%.4f, %.4f], mean %.4f, sd %.2e  (%d cells)\n" minimum(θ_int) maximum(θ_int) mean(θ_int) std(θ_int) length(intr)
        @printf "  interior wealth-independent (sd ≤ a few Δθ) : %s\n" (std(θ_int) <= 10Δθ)
        @printf "  gap (interior mean − formula) = %+.4f  (%.1f%% of formula)\n" (mean(θ_int) - θ_form) (100 * (mean(θ_int) - θ_form) / θ_form)
        @printf "  share-grid step Δθ            = %.4f\n" Δθ
    end

    return (; θ, interior = intr, θ_formula = θ_form, wgrid = collect(axis_grid(layout, :wealth)))
end

"""
Refine the wealth grid and watch the interior-mean seated θ* settle. The gap to
the formula has a fixed piece (the higher-moment / CRRA≈CARA error of the
two-point gamble, independent of `N_w`) plus a discretization piece (log-grid
interpolation of the continuation at off-node landings) that shrinks as `N_w`
grows — so the interior mean converges to the formula plus the residual
higher-moment correction. This is the discretization-gap demonstration.
"""
function cara_grid_refinement(p = cara_params; Ns = (80, 160, 320, 640), verbosity = 1)
    target = textbook_share(p)
    verbosity > 0 && @printf "Grid refinement (textbook θ* = %.4f; σ² = %.4f)\n" target p.d^2
    rows = NamedTuple[]
    for N in Ns
        res = cara_benchmark(CARANormalParams(p; N_w = N); verbosity = 0)
        m   = mean(res.θ[res.interior]); s = std(res.θ[res.interior])
        push!(rows, (; N_w = N, interior_mean = m, interior_sd = s, gap = m - target))
        verbosity > 0 && @printf "  N_w = %4d : interior mean θ* = %.4f, sd = %.2e, gap = %+.4f\n" N m s (m - target)
    end
    return rows
end

# Allow `CARANormalParams(p; field=…)` keyword-override construction.
CARANormalParams(p::CARANormalParams; kwargs...) =
    CARANormalParams(; p.γ, p.R_f, p.premium, p.d, p.p_up, p.shares, p.N_w, p.w_min, p.w_max, kwargs...)


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running CARA–normal portfolio benchmark…")
    @time cara_benchmark()
    println()
    cara_grid_refinement()
end
