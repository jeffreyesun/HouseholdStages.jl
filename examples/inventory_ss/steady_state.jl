################################################################
# (S,s) inventory management — steady state                       #
################################################################
#
# Partial equilibrium (exogenous price / discount rate): one stationary V/Λ solve over the joint
# (inventory, demand) state. The point to verify: the fixed reorder cost produces a LUMPY (S,s)
# policy — most firms sit in an inaction band (let the stock deplete, i' = s), a minority reorder up
# to a target — and a non-degenerate inventory cross-section. The reorder frequency is computed over
# the population FACING the choice: the stationary distribution pushed through the demand shock and
# the depletion drift, i.e. the post-shock, post-depletion stocks at which the reorder argmax fires.

include("model.jl")

using Printf

"""
Solve the (S,s) inventory stationary equilibrium and report: V finiteness, mass conservation, mean
inventory / sales / profit, the spread of the inventory marginal, the reorder frequency (fraction of
the post-shock, post-depletion population whose optimal target i' ≠ s — re-derived from the solved V),
and the extracted (S,s) target/trigger per demand state.
"""
function inventory_steady_state(p = inventory_params; verbosity = 1)
    firm = inventory_firm(p)
    env  = inventory_env(p)

    res = solve_steady_state_given_env!(firm, env)
    (; V, Λ, history) = res

    log_d, P_d = rouwenhorst(p.ρ_d, p.σ_d, p.N_d)
    d_grid = p.d_bar .* exp.(log_d)
    i_grid = collect(range(p.i_min, p.i_max; length = p.N_i))
    β      = 1 / (1 + p.r)

    m      = compute_moments(firm, Λ, env)
    i_marg = vec(sum(Λ; dims = 2))
    d_marg = vec(sum(Λ; dims = 1))

    # Re-derive the (S,s) reorder policy from the solved V. After demand d realizes and the stock
    # depletes to s, the firm picks i'* = argmax_{i'} [ M[i', s] + β·V(i', d) ] (the reorder argmax sits
    # inside the realized-demand continuation, shock-then-reorder timing). Reorder iff i'* ≠ s.
    M = [ji == ii ? 0.0 :
         ji  > ii ? -(p.F + p.c * (i_grid[ji] - i_grid[ii])) :
         -Inf
         for ji in 1:p.N_i, ii in 1:p.N_i]
    policy = zeros(Int, p.N_i, p.N_d)                       # policy[s, d] = chosen target index i'*
    for id in 1:p.N_d, is in 1:p.N_i
        best_val = -Inf; best_ji = is
        for ji in 1:p.N_i
            val = M[ji, is] + β * V[ji, id]
            val > best_val && (best_val = val; best_ji = ji)
        end
        policy[is, id] = best_ji
    end
    reorder = [policy[is, id] != is for is in 1:p.N_i, id in 1:p.N_d]

    # Population FACING the choice = stationary Λ pushed through (i) the demand shock and (ii) the
    # depletion drift, the two mass-moving stages before the reorder argmax. Λ is over end-states
    # (i', d); Λ_pre[s, d] is the post-shock, post-depletion mass at stock s in demand state d.
    Λ_shocked = Λ * P_d                                     # Λ_shocked[i, d] = Σ_{d_prev} Λ[i,d_prev]·P[d_prev,d]
    Λ_pre     = zeros(p.N_i, p.N_d)
    for id in 1:p.N_d
        S = depletion_matrix(i_grid, d_grid[id])            # S[i_from, i_to] one-hot draw-down
        Λ_pre[:, id] = S' * Λ_shocked[:, id]
    end
    reorder_freq = sum(Λ_pre[reorder]) / max(sum(Λ_pre), eps())

    # (S,s) bands per demand state: target S(d) = the reorder target from an empty/low stock; trigger
    # s(d) = the highest post-depletion stock that still triggers a reorder (the inaction-band floor).
    targets  = [i_grid[policy[1, id]] for id in 1:p.N_d]
    triggers = [let hits = findall(reorder[:, id]); isempty(hits) ? NaN : i_grid[maximum(hits)] end
                for id in 1:p.N_d]

    occupied = count(>(1e-10), i_marg)

    if verbosity > 0
        @printf "(S,s) inventory (price = %.2f, c = %.2f, h = %.2f, κ = %.2f, F = %.2f, r = %.3f)\n" p.price p.c p.h p.κ p.F p.r
        @printf "  grid: N_i = %d ∈ [%.1f, %.1f], N_d = %d, demand ∈ [%.2f, %.2f]\n" p.N_i p.i_min p.i_max p.N_d minimum(d_grid) maximum(d_grid)
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  V finite everywhere   = %s\n" all(isfinite, V)
        @printf "  ΣΛ (mass conserved)   = %.8f\n" sum(Λ)
        @printf "  mean inventory        = %.4f\n" m.mean_inventory
        @printf "  mean sales            = %.4f\n" m.mean_sales
        @printf "  mean profit           = %.4f\n" m.mean_profit
        @printf "  inventory marginal: %d / %d grid points carry mass\n" occupied p.N_i
        @printf "  reorder frequency     = %.4f  (fraction of post-depletion firms reordering; rest sit in inaction band)\n" reorder_freq
        @printf "  (S,s) target  S(d)    = [%s]\n" join((@sprintf("%.2f", x) for x in targets), ", ")
        @printf "  (S,s) trigger s(d)    = [%s]\n" join((isnan(x) ? "  NaN" : @sprintf("%.2f", x) for x in triggers), ", ")
    end

    return (; V, Λ, m, i_marg, d_marg, reorder_freq, targets, triggers, policy, reorder, Λ_pre, occupied, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving (S,s) inventory (Khan–Thomas) stationary equilibrium…")
    @time inventory_steady_state()
end
