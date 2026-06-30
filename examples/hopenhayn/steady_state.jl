######################################################
# Hopenhayn (1992) — stationary firm distribution      #
######################################################
#
# Partial equilibrium at a given wage `w`: one stationary V/Λ solve over the
# productivity state with entry and exit. Mass is NOT conserved (entrants in, exits
# out); the stationary firm mass settles at entrant-inflow / exit-rate. The free-entry
# condition and clearing that would pin `w` and the entrant mass `M` are the caller's
# OUTER LOOP — `free_entry_residual` (model.jl) is the scalar object to root on `w`.

include("model.jl")

using Printf

"""
Solve the Hopenhayn stationary firm distribution at the exogenous wage `p.w`. Runs one
`solve_steady_state_given_env!` over the (z, exiting) block, then reports: V finiteness,
total firm mass, the stationary exit rate (fraction of incumbents whose continuation
falls below scrap), mean productivity and employment, and the free-entry residual at a
sample entry cost (the scalar the outer loop would zero on `w`).
"""
function hopenhayn_steady_state(p = hopenhayn_params; c_e = 5.0, verbosity = 1)
    firm = hopenhayn_firm(p)
    env  = hopenhayn_env(p)

    res = solve_steady_state_given_env!(firm, env)
    (; V, Λ, history) = res

    m       = compute_moments(firm, Λ, env)
    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid  = exp.(log_z)
    ν       = invariant_dist(P_z)

    # V over z (the :exiting axis is size 1 at the output): V_start = V(z).
    V_z = vec(V)
    mass = m.mass

    # Exit rate in the cross-section: firms whose discounted continuation < scrap exit.
    # β·E[V(z')|z] is the value compared against scrap inside EndogenousExit.
    β     = 1 / (1 + p.r)
    contn = β .* (P_z * V_z)                         # β·E[V(z')|z] per z
    exits = contn .< p.scrap
    z_marg = vec(Λ)
    exit_rate = sum(z_marg[exits]) / max(sum(z_marg), eps())

    fe_resid = free_entry_residual(V_z, ν, c_e)

    if verbosity > 0
        @printf "Hopenhayn (1992): θ = %.2f, w = %.2f, c_f = %.2f, scrap = %.2f, r = %.3f\n" p.θ p.w p.c_f p.scrap p.r
        @printf "  grid: N_z = %d, ρ = %.2f, σ = %.2f\n" p.N_z p.ρ_z p.σ_z
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  V finite everywhere   = %s\n" all(isfinite, V)
        @printf "  firm mass             = %.4f\n" mass
        @printf "  exit rate             = %.4f\n" exit_rate
        @printf "  mean z (mass-wtd)     = %.4f\n" (mass > 0 ? m.mean_z / mass : NaN)
        @printf "  mean employment       = %.4f\n" (mass > 0 ? m.mean_emp / mass : NaN)
        @printf "  free-entry resid (c_e = %.1f) = %+.4f\n" c_e fe_resid
        @printf "  (resid > 0 ⇒ entry value exceeds c_e ⇒ wage too low: outer loop raises w)\n"
    end

    return (; V, Λ, m, mass, exit_rate, V_z, fe_resid, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Hopenhayn (1992) stationary firm distribution…")
    @time hopenhayn_steady_state()
end
