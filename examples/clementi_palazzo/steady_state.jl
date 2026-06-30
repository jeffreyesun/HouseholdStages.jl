######################################################################
# Clementi–Palazzo (2016) — stationary firm distribution               #
######################################################################
#
# Partial equilibrium at given prices: one stationary V/Λ solve over the joint
# `(k, z)` state with entry and endogenous exit. Mass is NOT conserved (entrants in,
# exits out); the stationary firm mass settles at entrant-inflow / exit-rate. The
# free-entry condition and clearing that would pin the price level and the entrant
# mass `M` — and the firm-size distribution as an aggregate state — are the caller's
# OUTER LOOP; `free_entry_residual` (model.jl) is the scalar object to root on.

include("model.jl")

using Printf

"""
Best continuation value `C(k,z) = max_{k'}[ −φ·max(k'−(1−δ)k,0)² + β·E_{z'|z}V(k',z') ]`
— the value the EndogenousExit stage compares against scrap (a firm exits when `C < scrap`).
Reconstructed directly from the solved value `V` (shape `(N_k, N_z)`) so the exit set is
computable in the cross-section, exactly replicating the CapitalInvestmentStage reward
(convex cost on gross investment, disinvestment free) and discount.
"""
function best_continuation(V_kz::AbstractMatrix, k_grid::AbstractVector, P_z::AbstractMatrix, p)
    β  = 1 / (1 + p.r)
    EV = V_kz * permutedims(P_z)               # EV[k',z] = Σ_z' P(z→z')·V(k',z')
    N_k, N_z = size(V_kz)
    C = similar(V_kz)
    for z in 1:N_z, ik in 1:N_k
        k = k_grid[ik]
        best = -Inf
        for jk in 1:N_k
            i    = max(k_grid[jk] - (1 - p.δ) * k, 0.0)
            cand = -p.φ * i^2 + β * EV[jk, z]
            cand > best && (best = cand)
        end
        C[ik, z] = best
    end
    return C
end

"""
Solve the Clementi–Palazzo stationary firm distribution at given prices. Runs one
`solve_steady_state_given_env!` over the `(k, z, exiting)` block, then reports: V
finiteness, total firm mass, the stationary exit rate (mass-weighted fraction of
incumbents whose best continuation falls below scrap, plus the mass-balance cross-check
entry-inflow / mass), mean capital, mass-weighted mean productivity vs the entrant mean
(survivor selection), the spread of the k-marginal, the capital-rises-in-z check, and the
free-entry residual at a sample entry cost.
"""
function clementi_palazzo_steady_state(p = clementi_palazzo_params; c_e = 5.0, verbosity = 1)
    firm = clementi_palazzo_firm(p)
    env  = clementi_palazzo_env(p)

    res = solve_steady_state_given_env!(firm, env)
    (; V, Λ, history) = res

    log_z, P_z = rouwenhorst(p.ρ_z, p.σ_z, p.N_z)
    z_grid     = exp.(log_z)
    ν          = invariant_dist(P_z)
    g          = entrant_distribution(ν, p.N_k)

    # Collapse the singleton :exiting axis ⇒ work over the (k, z) grid.
    V_kz   = reshape(V, p.N_k, p.N_z)
    Λ_kz   = reshape(Λ, p.N_k, p.N_z)
    k_grid = collect(range(p.k_min, p.k_max; length = p.N_k))

    m    = compute_moments(firm, Λ, env)
    mass = m.mass

    # Exit set in the cross-section: cells whose best continuation C(k,z) < scrap(k) = resale·k.
    # The distribution FACING the exit stage is the stationary Λ plus the just-arrived entrants
    # (entry then the identity-on-Λ profit stage precede exit), so the pre-exit mass at the
    # low-(k,z) exit cells includes the entrant duds who liquidate on arrival.
    C          = best_continuation(V_kz, k_grid, P_z, p)
    scrap      = p.resale .* k_grid                    # scrap(k), broadcast over z below
    exits      = C .< scrap                            # (N_k, N_z) Bool
    g_kz       = reshape(p.M .* g, p.N_k, p.N_z)       # entrant inflow on the (k, z) grid
    predist    = Λ_kz .+ g_kz                          # distribution the exit stage screens
    exit_flow  = sum(predist[exits])                   # mass that exits per period
    entry_flow = p.M * sum(g)                          # mass that enters per period
    exit_rate  = exit_flow / max(sum(Λ_kz), eps())     # exits per stationary incumbent
    # Mass-balance check: in steady state exit flow = entry flow (mass in = mass out).
    exit_rate_balance = entry_flow / max(mass, eps())

    # Survivor selection: mass-weighted mean z vs the entrant mean (∑ ν·z).
    z_marg        = vec(sum(Λ_kz; dims = 1))
    mean_z_surv   = mass > 0 ? m.mean_z / mass : NaN
    mean_z_entry  = sum(ν .* z_grid)

    # Capital marginal: spread + capital-rises-in-z policy profile.
    k_marg      = vec(sum(Λ_kz; dims = 2))
    occupied    = count(>(1e-10), k_marg)
    cdf         = cumsum(k_marg) ./ max(sum(k_marg), eps())
    q10         = k_grid[findfirst(>=(0.10), cdf)]
    q50         = k_grid[findfirst(>=(0.50), cdf)]
    q90         = k_grid[findfirst(>=(0.90), cdf)]
    mean_k_by_z = vec(sum(k_grid .* Λ_kz; dims = 1)) ./ max.(z_marg, eps())
    k_rising    = issorted(mean_k_by_z)

    fe_resid = free_entry_residual(V, g, c_e)

    if verbosity > 0
        @printf "Clementi–Palazzo (2016): α = %.2f, φ = %.1f, δ = %.2f, r = %.3f\n" p.α p.φ p.δ p.r
        @printf "  c_f = %.2f, scrap(k) = %.2f·k, entrant mass M = %.2f\n" p.c_f p.resale p.M
        @printf "  grid: N_k = %d ∈ [%.2f, %.2f], N_z = %d, ρ = %.2f, σ = %.2f\n" p.N_k p.k_min p.k_max p.N_z p.ρ_z p.σ_z
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  V finite everywhere      = %s\n" all(isfinite, V)
        @printf "  firm mass                = %.4f  (entry in, exit out — NOT conserved)\n" mass
        @printf "  exit flow vs entry flow  = %.4f vs %.4f   [mass balance: should match]\n" exit_flow entry_flow
        @printf "  exit rate (C < scrap)    = %.4f   [exits per stationary incumbent]\n" exit_rate
        @printf "  exit rate (mass balance) = %.4f   [entry inflow / mass — cross-check]\n" exit_rate_balance
        @printf "  mean k                   = %.4f\n" (mass > 0 ? m.mean_k / mass : NaN)
        @printf "  mean profit (z·k^α)      = %.4f\n" (mass > 0 ? m.mean_profit / mass : NaN)
        @printf "  mean z: survivors        = %.4f  vs entrants = %.4f  (selection ⇒ survivors > entrants)\n" mean_z_surv mean_z_entry
        @printf "  survivor selection holds = %s\n" (mean_z_surv > mean_z_entry)
        @printf "  k-marginal: %d / %d grid points carry mass\n" occupied p.N_k
        @printf "  k quantiles q10/q50/q90  = %.3f / %.3f / %.3f\n" q10 q50 q90
        @printf "  k NON-degenerate         = %s\n" (occupied > 1 && q90 > q10)
        @printf "  mean k by z              = [%s]\n" join((@sprintf("%.3f", x) for x in mean_k_by_z), ", ")
        @printf "  k rising in z            = %s\n" k_rising
        @printf "  free-entry resid (c_e = %.1f) = %+.4f\n" c_e fe_resid
        @printf "  (resid > 0 ⇒ entrant value exceeds c_e ⇒ price too favourable: outer loop adjusts)\n"
    end

    return (; V, Λ, m, mass, exit_flow, entry_flow, exit_rate, exit_rate_balance,
              mean_z_surv, mean_z_entry, mean_k_by_z, k_rising, occupied,
              q10, q50, q90, C, exits, fe_resid, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Clementi–Palazzo (2016) stationary firm distribution…")
    @time clementi_palazzo_steady_state()
end
