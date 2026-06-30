################################################################
# Lumpy / non-convex investment (Khan–Thomas 2008) — steady state #
################################################################
#
# Partial equilibrium (exogenous discount rate): one stationary V/Λ solve over the joint
# (k, z) state. The point to verify: the fixed cost produces a LUMPY adjustment policy —
# most firms sit in an inaction band (keep k' = k), a minority adjust in bursts — and a
# non-degenerate capital cross-section that rises in productivity z.

include("model.jl")

using Printf

"""
Solve the lumpy-investment stationary equilibrium and report: V finiteness, mass
conservation, mean capital / profit / z, the spread of the k-marginal, the
adjustment frequency (fraction of mass at (k, z) whose optimal k' ≠ k — re-derived
from the solved V), and the monotonicity of mean capital in z.
"""
function lumpy_investment_steady_state(p = lumpy_investment_params; verbosity = 1)
    firm = lumpy_investment_firm(p)
    env  = lumpy_investment_env(p)

    res = solve_steady_state_given_env!(firm, env)
    (; V, Λ, history) = res

    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid = exp.(log_z)
    k_grid = collect(range(p.k_min, p.k_max; length = p.N_k))
    β      = 1 / (1 + p.r)

    m      = compute_moments(firm, Λ, env)
    k_marg = vec(sum(Λ; dims = 2))
    z_marg = vec(sum(Λ; dims = 1))
    mean_k_by_z = vec(sum(k_grid .* Λ; dims = 1)) ./ max.(z_marg, eps())

    # Re-derive the lumpy policy from the solved V. The argmax sits INSIDE the z-expectation
    # (shock-then-invest timing), so for state (k, realized z) the firm picks
    # k'* = argmax_{k'} [ −F·1{k'≠k} + β·V(k', z) ]. Adjust if k'* ≠ k. Λ is over end-states
    # (k', z'); the adjustment frequency is its Λ-weighted average (a cross-sectional proxy).
    adjust = falses(p.N_k, p.N_z)
    for iz in 1:p.N_z, ik in 1:p.N_k
        best_val = -Inf; best_ik = ik
        for jk in 1:p.N_k
            cost = (jk == ik) ? 0.0 : -p.F
            val  = cost + β * V[jk, iz]
            val > best_val && (best_val = val; best_ik = jk)
        end
        adjust[ik, iz] = (best_ik != ik)
    end
    # The population FACING the choice is post-shock, pre-investment: transition the stationary Λ
    # through the z-shock (k unchanged), then weight the adjust mask by it.
    Λ_pre = Λ * P_z                                          # Λ_pre[k, z'] = Σ_z Λ[k,z]·P[z,z']
    adj_freq = sum(Λ_pre[adjust]) / max(sum(Λ_pre), eps())

    occupied = count(>(1e-10), k_marg)

    if verbosity > 0
        @printf "Lumpy investment, Khan–Thomas (α = %.2f, δ = %.2f, F = %.2f, r = %.3f)\n" p.α p.δ p.F p.r
        @printf "  grid: N_k = %d ∈ [%.1f, %.1f], N_z = %d\n" p.N_k p.k_min p.k_max p.N_z
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  V finite everywhere   = %s\n" all(isfinite, V)
        @printf "  ΣΛ (mass conserved)   = %.8f\n" sum(Λ)
        @printf "  mean k                = %.4f\n" m.mean_k
        @printf "  mean profit (z·k^α)   = %.4f\n" m.mean_profit
        @printf "  k-marginal: %d / %d grid points carry mass\n" occupied p.N_k
        @printf "  adjustment frequency  = %.4f  (fraction of firms re-tuning k; rest sit in inaction band)\n" adj_freq
        @printf "  mean k by z           = [%s]\n" join((@sprintf("%.2f", x) for x in mean_k_by_z), ", ")
        @printf "  k rising in z         = %s\n" issorted(mean_k_by_z)
    end

    return (; V, Λ, m, k_marg, z_marg, mean_k_by_z, adj_freq, occupied, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving lumpy-investment (Khan–Thomas) stationary equilibrium…")
    @time lumpy_investment_steady_state()
end
