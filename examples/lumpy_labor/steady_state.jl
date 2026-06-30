################################################################
# Lumpy / (S,s) labor adjustment (Cooper–Haltiwanger–Willis) — steady state #
################################################################
#
# Partial equilibrium (exogenous wage / discount rate): one stationary V/Λ solve over the joint
# (n, z) state. The point to verify: the fixed cost produces a LUMPY adjustment policy — most
# firms sit in an inaction band (keep n' = n) and a minority hire/fire in bursts — and a
# non-degenerate employment cross-section that rises in productivity z.

include("model.jl")

using Printf

"""
Solve the lumpy-labor stationary equilibrium and report: V finiteness, mass conservation, mean
employment / profit / z, the spread of the n-marginal, the adjustment frequency (fraction of
mass at (n, z) whose optimal n' ≠ n — re-derived from the solved V), and the monotonicity of
mean employment in z.
"""
function lumpy_labor_steady_state(p = lumpy_labor_params; verbosity = 1)
    firm = lumpy_labor_firm(p)
    env  = lumpy_labor_env(p)

    res = solve_steady_state_given_env!(firm, env)
    (; V, Λ, history) = res

    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid = exp.(log_z)
    n_grid = collect(range(p.n_min, p.n_max; length = p.N_n))
    β      = 1 / (1 + p.r)

    m      = compute_moments(firm, Λ, env)
    n_marg = vec(sum(Λ; dims = 2))
    z_marg = vec(sum(Λ; dims = 1))
    mean_n_by_z = vec(sum(n_grid .* Λ; dims = 1)) ./ max.(z_marg, eps())

    # Re-derive the lumpy policy from the solved V. The argmax sits INSIDE the z-expectation
    # (shock-then-adjust timing), so for state (n, realized z) the firm picks
    # n'* = argmax_{n'} [ −F·1{n'≠n} + β·V(n', z) ]. Adjust if n'* ≠ n.
    adjust = falses(p.N_n, p.N_z)
    for iz in 1:p.N_z, in in 1:p.N_n
        best_val = -Inf; best_in = in
        for jn in 1:p.N_n
            cost = (jn == in) ? 0.0 : -p.F
            val  = cost + β * V[jn, iz]
            val > best_val && (best_val = val; best_in = jn)
        end
        adjust[in, iz] = (best_in != in)
    end
    # The population FACING the choice is post-shock, pre-adjust: transition the stationary Λ
    # through the z-shock (n unchanged), then weight the adjust mask by it.
    Λ_pre = Λ * P_z                                          # Λ_pre[n, z'] = Σ_z Λ[n,z]·P[z,z']
    adj_freq = sum(Λ_pre[adjust]) / max(sum(Λ_pre), eps())

    occupied = count(>(1e-10), n_marg)

    if verbosity > 0
        @printf "Lumpy labor, Cooper–Haltiwanger–Willis (θ = %.2f, w = %.2f, F = %.2f, r = %.3f)\n" p.θ p.w p.F p.r
        @printf "  grid: N_n = %d ∈ [%.2f, %.1f], N_z = %d\n" p.N_n p.n_min p.n_max p.N_z
        @printf "  frictionless target n*(z): [%s]\n" join((@sprintf("%.2f", n_star(z, p)) for z in z_grid), ", ")
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  V finite everywhere   = %s\n" all(isfinite, V)
        @printf "  ΣΛ (mass conserved)   = %.8f\n" sum(Λ)
        @printf "  mean n                = %.4f\n" m.mean_n
        @printf "  mean profit (z·n^θ−wn)= %.4f\n" m.mean_profit
        @printf "  n-marginal: %d / %d grid points carry mass\n" occupied p.N_n
        @printf "  adjustment frequency  = %.4f  (fraction hiring/firing; rest sit in inaction band)\n" adj_freq
        @printf "  mean n by z           = [%s]\n" join((@sprintf("%.2f", x) for x in mean_n_by_z), ", ")
        @printf "  n rising in z         = %s\n" issorted(mean_n_by_z)
    end

    return (; V, Λ, m, n_marg, z_marg, mean_n_by_z, adj_freq, occupied, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving lumpy-labor (Cooper–Haltiwanger–Willis) stationary equilibrium…")
    @time lumpy_labor_steady_state()
end
