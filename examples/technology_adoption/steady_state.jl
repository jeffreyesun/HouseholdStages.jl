######################################################################
# Technology adoption — network-externality fixed point (outer loop)  #
######################################################################

# The household block (adopt choice) is exercised inside an OUTER loop that
# closes the NETWORK EXTERNALITY: the adopted payoff `θ·share` depends on the
# aggregate adoption share, and the share is the stationary mass of adopters.
# We iterate `share ← mass(adopted)` to a fixed point. Because the feedback
# is positive, the model can have MULTIPLE fixed points — so we solve from a
# LOW and a HIGH initial share to expose any low-adoption trap vs.
# high-adoption equilibrium. The per-share V/Λ solve is delegated to
# `solve_steady_state_given_env!`; the share iteration is here.

include("model.jl")

using Printf

"""
Iterate the adoption-share fixed point `share = mass(adopted)` from a given
initial share, damped, until it stops moving.
"""
function adoption_fixed_point(share0::Real, p = params;
                              damping = 0.5, tol = 1e-5, maxiter = 500)
    hh = tech_adopt_household(p)
    share = float(share0)
    V, Λ = nothing, nothing
    iters = 0
    while iters < maxiter
        env = tech_adopt_env(share)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        (; V, Λ) = res
        share_new = compute_moments(hh, Λ, env).adoption_share
        iters += 1
        abs(share_new - share) < tol && (share = share_new; break)
        share = (1 - damping) * share + damping * share_new
    end
    return (; share, iters)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving technology-adoption network-externality fixed point…")
    @time begin
        lo = adoption_fixed_point(0.02)
        hi = adoption_fixed_point(0.98)
    end
    @printf "  from LOW  initial share (0.02): converged share = %.4f  (%d iters)\n" lo.share lo.iters
    @printf "  from HIGH initial share (0.98): converged share = %.4f  (%d iters)\n" hi.share hi.iters
    if abs(lo.share - hi.share) > 1e-3
        println("  ⇒ MULTIPLE equilibria: a low-adoption trap and a high-adoption equilibrium coexist.")
    else
        println("  ⇒ Unique adoption equilibrium at these parameters (both starts converge to it).")
    end
end
