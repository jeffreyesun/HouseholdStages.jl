###############################################################
# AR(1) income steady state — single solve at fixed exogenous r #
###############################################################
#
# Fixed-r partial-equilibrium self-insurance solve (the Bewley framing):
# `r` is exogenous and below `1/β − 1`, so there is no market to clear and
# the whole solve is a single inner V/Λ fixed point at the given env. The
# only thing distinguishing this from `examples/bewley` is the income
# process — a 7-state Rouwenhorst discretization of a log-AR(1) rather than
# a hand-written 3-state chain. That difference is resolved offline, before
# `solve_steady_state_given_env!` ever runs.

include("model.jl")

using Printf

"""
Solve the AR(1)-income self-insurance steady state at the fixed exogenous
return `r` (one inner V/Λ fixed-point solve, no market clearing) and report
the aggregate buffer stock and the discretized income process. `method`
selects the offline discretization (`:rouwenhorst` default, `:tauchen`).
"""
function income_ar1_steady_state(p = income_ar1_params; method = :rouwenhorst, verbosity = 1)
    hh  = income_ar1_household(p; method)
    env = income_ar1_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    y_grid, T = method === :tauchen ? tauchen(p.ρ, p.σ_ε, p.N_y) :
                                      rouwenhorst(p.ρ, p.σ_ε, p.N_y)

    if verbosity > 0
        @printf "AR(1) income steady state — %s discretization (ρ = %.2f, σ_ε = %.2f, N_y = %d)\n" method p.ρ p.σ_ε p.N_y
        @printf "  income grid : %s\n" string(round.(y_grid; digits = 3))
        @printf "  r = %.4f   (1/β − 1 = %.4f, impatience gap = %.4f)\n" p.r (1/p.β - 1) (1/p.β - 1 - p.r)
        @printf "  ΣΛ          = %.6f\n" sum(res.Λ)
        @printf "  A_mean      = %.4f\n" m.A_mean
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end

    return (; r = p.r, V = res.V, Λ = res.Λ, A_mean = m.A_mean,
              y_grid, T, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving AR(1)-income self-insurance steady state…")
    @time res = income_ar1_steady_state()
    @printf "  ΣΛ = %.6f, A_mean = %.4f\n" sum(res.Λ) res.A_mean
end
