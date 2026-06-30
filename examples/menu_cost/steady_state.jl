################################################################
# Menu-cost price setting (Golosov–Lucas / Nakamura–Steinsson) — steady state #
################################################################
#
# Partial equilibrium at a fixed inflation rate π (the monetary block is the caller's outer loop):
# one stationary V/Λ solve over the joint (price, z) state. The points to verify: the menu cost
# produces an (S,s) PRICING BAND — most firms KEEP their price (sit in the inaction band, letting it
# erode), a minority RESET in bursts — a sensible FREQUENCY OF PRICE CHANGE (the Nakamura–Steinsson
# moment), and a non-degenerate cross-section of relative prices.

include("model.jl")

using Printf

"""
Solve the menu-cost stationary equilibrium and report: V finiteness, mass conservation, mean price /
profit / marginal cost, the spread of the price-marginal, and the FREQUENCY OF PRICE CHANGE — the
fraction of the post-shock, post-erosion population whose optimal price p' ≠ p (re-derived from the
solved V, the Nakamura–Steinsson moment).
"""
function menu_cost_steady_state(p = menu_cost_params; verbosity = 1)
    firm = menu_cost_firm(p)
    env  = menu_cost_env(p)

    res = solve_steady_state_given_env!(firm, env)
    (; V, Λ, history) = res

    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid = exp.(log_z)
    p_grid = price_grid(p)
    β      = 1 / (1 + p.r)

    m      = compute_moments(firm, Λ, env)
    p_marg = vec(sum(Λ; dims = 2))
    z_marg = vec(sum(Λ; dims = 1))
    mean_p_by_z = vec(sum(p_grid .* Λ; dims = 1)) ./ max.(z_marg, eps())

    # Re-derive the (S,s) reset policy from the solved V. The reset argmax sits just inside the discount,
    # so for an incoming price index ip and marginal cost iz the firm picks
    # p'* = argmax_{p'} [ −F·1{p'≠ip} + β·V(p', iz) ]. It RESETS if p'* ≠ ip.
    reset = falses(p.N_p, p.N_z)
    for iz in 1:p.N_z, ip in 1:p.N_p
        best_val = -Inf; best_jp = ip
        for jp in 1:p.N_p
            cost = (jp == ip) ? 0.0 : -p.F
            val  = cost + β * V[jp, iz]
            val > best_val && (best_val = val; best_jp = jp)
        end
        reset[ip, iz] = (best_jp != ip)
    end
    # The population FACING the reset choice is post-shock, post-erosion: push the stationary Λ forward
    # through the z-shock (Λ·P_z), then through the deterministic price down-shift, then weight the reset mask.
    Λ_shock = Λ * P_z                                       # Λ_shock[p, z'] = Σ_z Λ[p,z]·P[z,z']
    Λ_pre   = zeros(p.N_p, p.N_z)                           # erosion: each price slides down one index
    for iz in 1:p.N_z, ip in 1:p.N_p
        Λ_pre[max(ip - 1, 1), iz] += Λ_shock[ip, iz]
    end
    freq_change = sum(Λ_pre[reset]) / max(sum(Λ_pre), eps())

    occupied = count(>(1e-10), p_marg)
    mean_markup = m.mean_price / max(m.mean_z, eps())

    if verbosity > 0
        @printf "Menu-cost pricing, Golosov–Lucas / Nakamura–Steinsson (ε = %.1f, F = %.4f, π = %.3f, r = %.3f)\n" p.ε p.F p.π p.r
        @printf "  grid: N_p = %d ∈ [%.2f, %.2f] (log step = log(1+π)), N_z = %d\n" p.N_p first(p_grid) last(p_grid) p.N_z
        @printf "  frictionless markup ε/(ε−1) = %.3f\n" p.ε / (p.ε - 1)
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  V finite everywhere     = %s\n" all(isfinite, V)
        @printf "  ΣΛ (mass conserved)     = %.8f\n" sum(Λ)
        @printf "  mean relative price     = %.4f\n" m.mean_price
        @printf "  mean marginal cost z    = %.4f\n" m.mean_z
        @printf "  mean realized markup    = %.4f  (mean p / mean z)\n" mean_markup
        @printf "  mean profit             = %.4f\n" m.mean_profit
        @printf "  price-marginal: %d / %d grid points carry mass\n" occupied p.N_p
        @printf "  FREQUENCY OF PRICE CHANGE = %.4f  (fraction resetting per period; rest sit in the band)\n" freq_change
        @printf "  mean price by z         = [%s]\n" join((@sprintf("%.3f", x) for x in mean_p_by_z), ", ")
        @printf "  reset target rises in z = %s\n" issorted(mean_p_by_z)
    end

    return (; V, Λ, m, p_marg, z_marg, mean_p_by_z, freq_change, occupied, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving menu-cost (Golosov–Lucas / Nakamura–Steinsson) stationary equilibrium…")
    @time menu_cost_steady_state()
end
