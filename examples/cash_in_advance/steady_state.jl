###############################################################
# CIA steady state — single fixed-(r,P) inner solve            #
###############################################################
#
# Cash-in-advance is a partial-equilibrium experiment at fixed real return
# `r` and price level `P`: one `solve_steady_state_given_env!` delivers the
# stationary money distribution. The block (`model.jl`) is the Aiyagari
# spine; the Clower content is the `c ≤ m/P` mask inside the savings utility
# closure.

include("model.jl")

using Printf

"""
Solve the CIA household at the fixed env `(r, w, P)`: one inner V/Λ solve.
Reports the share of households whose cash constraint binds (those that
consume essentially all real money on hand), alongside the converged
`V`/`Λ` and aggregate money holdings.
"""
function cia_steady_state(p = cia_params; verbosity = 1)
    hh  = cia_household(p)
    env = cia_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    M = compute_moments(hh, Λ, env).M_supplied

    if verbosity > 0
        @printf "CIA steady state (β = %.2f, σ = %.1f, r = %.3f, P = %.2f)\n" p.β p.σ p.r p.P
        @printf "  ΣΛ (total mass)     : %.6f\n" sum(Λ)
        @printf "  V finite            : %s\n" all(isfinite, V)
        @printf "  aggregate money M   : %.4f\n" M
        @printf "  VFI iters / Λ iters : %d / %d\n" history.vfi_iters history.lambda_iters
    end

    return (; V, Λ, M, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving cash-in-advance steady state…")
    @time cia_steady_state()
end
