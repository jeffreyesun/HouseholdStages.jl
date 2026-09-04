######################################################################
# Becker–Tomes — stationary dynastic steady state                     #
######################################################################

# A dynasty is recursive with the SAME value function each generation, so the
# steady state IS the cross-generation fixed point: `solve_steady_state_given_env!`
# finds the dynastic value `V(b,h)` (VFI) and the stationary distribution of
# dynasties `Λ(b,h)`. No custom cohort iteration — the recursion is the steady
# state. Partial equilibrium (r, w exogenous). Intergenerational persistence is
# read off the seated policy: a parent at human capital `h` is pushed one
# generation forward and we read the mean child human capital `E[h' | h]`.

include("model.jl")

using Printf

"""
Push a single dynasty starting at `(b, h)` (a point mass at the nearest grid cell)
one generation forward through the SEATED policy, and return the mean child wealth
and mean child human capital. Used to trace intergenerational persistence
`E[(b',h') | (b,h)]`. Assumes the policy is already seated for `env` (call
`backward!` at the converged `V` first).
"""
function bt_next_generation(hh, b::Real, h::Real, bgrid, hgrid, sz)
    ib = argmin(abs.(bgrid .- b)); ih = argmin(abs.(hgrid .- h))
    Λ  = zeros(sz); Λ[ib, ih, 1] = 1.0
    Λ′ = forward!(hh, Λ)
    m  = sum(Λ′)
    cells = cell_array(end_layout(hh))
    mean_b = sum(getproperty.(cells, :wealth) .* Λ′) / m
    mean_h = sum(getproperty.(cells, :h)      .* Λ′) / m
    return (; mean_b, mean_h)
end

"""
Solve the Becker–Tomes dynasty as a stationary steady state, then trace
intergenerational persistence. Returns the dynastic value/distribution, the
aggregate moments, and `E[h' | h]` at low/median/high parental human capital plus
the implied intergenerational HC elasticity.
"""
function becker_tomes_steady_state(p = becker_tomes_params; verbosity = 1)
    hh   = becker_tomes_household(p)
    env  = (; r = p.r, w = p.w)
    res  = solve_steady_state_given_env!(hh, env)
    m    = compute_moments(hh, res.Λ, env)

    bgrid = collect(Float64, axisvalues(GriddedContinuous(0.0, p.b_max, p.N_b; spacing = :log)))
    hgrid = collect(range(p.h_min, p.h_max; length = p.N_h))
    sz    = layout_size(start_layout(hh))

    # Re-seat the converged policy (stationary ⇒ continuation V_end = V_start), then
    # trace the cross-generation HC map at a fixed (median) bequest.
    backward!(hh, res.V, env)
    b_med = bgrid[cld(p.N_b, 2)]
    h_lo, h_md, h_hi = hgrid[cld(p.N_h, 4)], hgrid[cld(p.N_h, 2)], hgrid[3 * p.N_h ÷ 4]
    nlo = bt_next_generation(hh, b_med, h_lo, bgrid, hgrid, sz)
    nmd = bt_next_generation(hh, b_med, h_md, bgrid, hgrid, sz)
    nhi = bt_next_generation(hh, b_med, h_hi, bgrid, hgrid, sz)
    # Intergenerational HC elasticity from the low/high endpoints (log-log slope).
    ig_elasticity = (log(nhi.mean_h) - log(nlo.mean_h)) / (log(h_hi) - log(h_lo))
    persistence_ok = nlo.mean_h < nmd.mean_h < nhi.mean_h

    if verbosity > 0
        @printf "Becker–Tomes dynastic steady state (σ = %.1f, α = %.2f, θ = %.2f, ψ = %.2f)\n" p.σ p.α p.θ p.ψ
        @printf "  mass(Λ)                        = %.6f  (fertility = %.2f)\n" sum(res.Λ) p.fertility
        @printf "  V finite everywhere            = %s\n" all(isfinite, res.V)
        @printf "  VFI iters / Λ iters            = %d / %d\n" res.history.vfi_iters res.history.lambda_iters
        @printf "  mean wealth (bequest)          = %.4f\n" m.mean_wealth
        @printf "  mean human capital             = %.4f\n" m.mean_h
        @printf "  E[h' | h] at h = %.2f / %.2f / %.2f = %.3f / %.3f / %.3f\n" h_lo h_md h_hi nlo.mean_h nmd.mean_h nhi.mean_h
        @printf "  intergenerational persistence  = %s\n" persistence_ok
        @printf "  intergenerational HC elasticity= %.3f\n" ig_elasticity
    end
    return (; V = res.V, Λ = res.Λ, m.mean_wealth, m.mean_h,
              persistence_ok, ig_elasticity, history = res.history)
end


if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Becker–Tomes dynastic steady state…")
    @time becker_tomes_steady_state()
end
