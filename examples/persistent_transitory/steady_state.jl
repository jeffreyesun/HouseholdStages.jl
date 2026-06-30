###################################################################
# Persistent+transitory steady state — single solve at fixed r     #
###################################################################
#
# Fixed-r partial-equilibrium self-insurance solve. The novelty over the
# single-shock examples is entirely in the household block (two income
# axes); the outer loop is the same single inner V/Λ fixed point.

include("model.jl")

using Printf

"""
Solve the persistent+transitory self-insurance steady state at the fixed
exogenous return `r` (one inner V/Λ fixed point) and report the aggregate
buffer stock and the two income components.
"""
function persistent_transitory_steady_state(p = persistent_transitory_params; verbosity = 1)
    hh  = persistent_transitory_household(p)
    env = persistent_transitory_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    if verbosity > 0
        @printf "Persistent+transitory income steady state\n"
        @printf "  persistent grid : %s\n" string(p.z_grid)
        @printf "  transitory grid : %s  (iid, row %s)\n" string(p.ν_grid) string(p.p_ν)
        @printf "  r = %.4f   (1/β − 1 = %.4f, impatience gap = %.4f)\n" p.r (1/p.β - 1) (1/p.β - 1 - p.r)
        @printf "  ΣΛ      = %.6f\n" sum(res.Λ)
        @printf "  A_mean  = %.4f\n" m.A_mean
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end

    return (; r = p.r, V = res.V, Λ = res.Λ, A_mean = m.A_mean, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving persistent+transitory self-insurance steady state…")
    @time res = persistent_transitory_steady_state()
    @printf "  ΣΛ = %.6f, A_mean = %.4f\n" sum(res.Λ) res.A_mean
end
