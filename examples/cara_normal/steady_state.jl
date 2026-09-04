####################################################################
# CARA–normal benchmark — seated θ*(x) vs the textbook formula      #
####################################################################

# Not a steady state but a ONE-PERIOD mean-variance check (the file keeps the
# examples' conventional name). With terminal value `V_end(w) = u_crra(w)`, one
# `backward!` of the `GaussianLoadingStage` seats the static optimal share `θ*(x)`
# per wealth cell. The check: θ*(x) is interior and (for CRRA) wealth-INDEPENDENT
# — flat across the grid INTERIOR — and close to the closed-form
# `θ* = (μ − R_f)/(γσ²)` up to (i) the log-grid interpolation of the
# continuation at the off-node landings and (ii) the local CRRA≈CARA /
# higher-moment error (the formula is a 2nd-order mean-variance approximation
# of the CRRA objective). Neither gap is a discretization of the share: θ is
# continuous, and the returns are (±8σ-truncated) Gaussian, so the benchmark's
# distributional assumption holds by construction.
#
# Grid EDGES are excluded: a landing `w·(R_f + θ·(μ_x + σ_x·Z))` from a
# near-edge cell spills outside `[w_min, w_max]` and is clamped, breaking the
# exact homogeneity that makes the CRRA share wealth-independent. The check is
# therefore stated on the INTERIOR band — cells whose landing band (out to a
# few sd) stays on-grid.

include("model.jl")

using Printf, Statistics

"""
Wealth-grid indices whose worst-case (max-share, −4σ) and best-case (max-share,
+4σ) landings both stay within `[w_min, w_max]` — the interior band on which the
homogeneity (hence the wealth-independent CRRA share) is not corrupted by
landing clamping at the grid edges. The Gaussian row extends to ±8σ, but the
mass beyond ±4σ (< 1e-4) is below the check's resolution, so 4σ bounds the band
without emptying the interior.
"""
function cara_interior(layout, p)
    w      = collect(axis_grid(layout, :wealth))
    μx, σx = excess_moments(p)
    smx    = last(p.share_bounds)
    den_lo = p.R_f + smx * (μx - 4σx)
    lo     = den_lo > 0 ? p.w_min / den_lo : p.w_min   # a negative worst-case landing clamps at the floor with negligible mass
    hi     = p.w_max / (p.R_f + smx * (μx + 4σx))
    return findall(x -> lo <= x <= hi, w)
end

"""
Run the one-period CARA–normal benchmark: build the `GaussianLoadingStage`, drive a
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
    tol    = 0.05                                      # flatness tolerance (kept from the θ-grid era's 10Δθ)

    if verbosity > 0
        μx, σx = excess_moments(p)
        @printf "CARA–normal one-period benchmark (γ = %.1f, μ = %.3f, R_f = %.3f, σ² = %.4f)\n" p.γ (p.R_f + μx) p.R_f σx^2
        @printf "  textbook θ* = (μ−R_f)/(γσ²) = %.4f\n" θ_form
        @printf "  seated θ*(x), full grid : [%.4f, %.4f]  (edges clamp to the corners)\n" minimum(θ) maximum(θ)
        @printf "  seated θ*(x), interior  : [%.4f, %.4f], mean %.4f, sd %.2e  (%d cells)\n" minimum(θ_int) maximum(θ_int) mean(θ_int) std(θ_int) length(intr)
        @printf "  interior wealth-independent (sd ≤ %.2f) : %s\n" tol (std(θ_int) <= tol)
        @printf "  gap (interior mean − formula) = %+.4f  (%.1f%% of formula)\n" (mean(θ_int) - θ_form) (100 * (mean(θ_int) - θ_form) / θ_form)
    end

    return (; θ, interior = intr, θ_formula = θ_form, wgrid = collect(axis_grid(layout, :wealth)))
end

"""
Refine the wealth grid and watch the interior-mean seated θ* settle. The gap to
the formula has a fixed piece (the CRRA≈CARA higher-moment error, independent
of `N_w` — the returns themselves are Gaussian) plus a
discretization piece (log-grid interpolation of the continuation at off-node
landings) that shrinks as `N_w` grows — so the interior mean converges to the
formula plus the residual higher-moment correction. This is the
discretization-gap demonstration.
"""
function cara_grid_refinement(p = cara_params; Ns = (80, 160, 320, 640), verbosity = 1)
    target = textbook_share(p)
    verbosity > 0 && @printf "Grid refinement (textbook θ* = %.4f; σ² = %.4f)\n" target excess_moments(p)[2]^2
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
    CARANormalParams(; p.γ, p.R_f, p.premium, p.d, p.p_up, p.share_bounds, p.N_w, p.w_min, p.w_max, kwargs...)


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running CARA–normal portfolio benchmark…")
    @time cara_benchmark()
    println()
    cara_grid_refinement()
end
