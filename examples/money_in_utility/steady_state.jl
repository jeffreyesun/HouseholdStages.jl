###############################################################
# MIU steady state — single fixed-(r,P) inner solve            #
###############################################################
#
# Money-in-the-utility is a partial-equilibrium experiment at fixed real
# return `r` and price level `P`: there is no market to clear, so the
# stationary distribution follows from ONE `solve_steady_state_given_env!`.
# The household block (`model.jl`) is the Aiyagari spine; the Sidrauski
# content is the real-balance term inside the savings utility closure.

include("model.jl")

using Printf

"""
Solve the MIU household at the fixed env `(r, w, P)`: one inner V/Λ solve.
Returns the converged `V`, `Λ`, aggregate real balances `M`, and mean
consumption, plus the iteration counts.
"""
function miu_steady_state(p = miu_params; verbosity = 1)
    hh  = miu_household(p)
    env = miu_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    M = compute_moments(hh, Λ, env).M_supplied

    if verbosity > 0
        @printf "MIU steady state (β = %.2f, σ = %.1f, σ_m = %.1f, χ = %.2f, r = %.3f, P = %.2f)\n" p.β p.σ p.σ_m p.χ p.r p.P
        @printf "  ΣΛ (total mass)         : %.6f\n" sum(Λ)
        @printf "  V finite                : %s\n" all(isfinite, V)
        @printf "  aggregate real balances : %.4f\n" M
        @printf "  VFI iters / Λ iters     : %d / %d\n" history.vfi_iters history.lambda_iters
    end

    return (; V, Λ, M, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving money-in-the-utility steady state…")
    @time miu_steady_state()
end
