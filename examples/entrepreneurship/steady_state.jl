######################################################################
# Entrepreneurship steady state — occupational sorting & wealth tail   #
######################################################################

# A single fixed-price solve: returns, productivity, and the wage are
# exogenous, so there is no outer tatonnement. The household block is
# `OccChoice ∘ ⊕_occupation{worker, entrepreneur}` — library stages only, no
# bespoke household stage (see `model.jl` for why the legs are forced to be
# structurally uniform). The report shows the Quadrini / Cagetti–De Nardi
# pattern: households sort into entrepreneurship by wealth and productivity, and
# the entrepreneurial return premium fattens the top wealth tail.

include("model.jl")

using Printf

"""
Solve the entrepreneurship household at the fixed env and report occupational
sorting and wealth concentration. The `:occupation` axis (position 3 of `Λ`)
slices worker (1) vs entrepreneur (2). Returns `(V, Λ)`, the aggregate
`mean_wealth`, the entrepreneur population share, the per-occupation mean
wealth, and the top-1%/top-10% wealth shares.
"""
function entrepreneurship_steady_state(p = entrepreneurship_params; a_floor = p.a_floor, verbosity = 1)
    hh  = entrepreneurship_household(p)
    env = entrepreneurship_env(p; a_floor)
    res = solve_steady_state_given_env!(hh, env)
    Λ   = res.Λ
    m   = compute_moments(hh, Λ, env)

    # Per-occupation decomposition from the stationary Λ (occupation at position 3).
    out_layout = hh.buffer.output_layout
    cells      = cell_array(out_layout)
    wealth     = getproperty.(cells, :wealth)

    occ_mass        = [sum(Λ[:, :, o])                          for o in 1:2]
    occ_wealth_sum  = [sum(wealth[:, :, o] .* Λ[:, :, o])       for o in 1:2]
    occ_mean_wealth = occ_wealth_sum ./ max.(occ_mass, eps())
    entre_share     = occ_mass[2]                                # mass already sums to 1

    # Wealth concentration: top-1% and top-10% shares of total wealth.
    w_flat = vec(wealth); λ_flat = vec(Λ)
    ord    = sortperm(w_flat)
    w_s, λ_s = w_flat[ord], λ_flat[ord]
    cum_mass = cumsum(λ_s)
    total_wealth = sum(w_s .* λ_s)
    top_share(q) = sum((w_s .* λ_s)[cum_mass .>= 1 - q]) / total_wealth
    top1, top10 = top_share(0.01), top_share(0.10)

    if verbosity > 0
        meanR = p.p_up * p.R_up + (1 - p.p_up) * p.R_dn
        @printf "Entrepreneurship steady state (σ = %.1f, E[R_business] = %.3f vs R_f = %.2f, κ = %.2f)\n" p.σ meanR p.R_f p.κ
        @printf "  mass(Λ)               = %.6f\n"  sum(Λ)
        @printf "  aggregate mean wealth = %.4f\n"  m.mean_wealth
        @printf "  entrepreneur share    = %.3f  (moment: %.3f)\n" entre_share m.entrepreneur_share
        @printf "  mean wealth: worker   = %.4f\n"  occ_mean_wealth[1]
        @printf "  mean wealth: entrep.  = %.4f  (%.1f× the worker)\n" occ_mean_wealth[2] (occ_mean_wealth[2] / max(occ_mean_wealth[1], eps()))
        @printf "  top-1%%  wealth share  = %.3f\n" top1
        @printf "  top-10%% wealth share  = %.3f\n" top10
        @printf "  VFI iters = %d, Λ iters = %d\n"  res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ, mean_wealth = m.mean_wealth, entre_share,
              occ_mean_wealth, top1, top10, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving entrepreneurship (occupational-choice) steady state…")
    @time res = entrepreneurship_steady_state()

    # Comparative static: a steeper business success multiple pulls more
    # households into entrepreneurship; the wider entry slightly dilutes the
    # top-10% concentration (more, less-selected entrants).
    println("\nComparative static — entrepreneur share & top-10% share vs business success multiple R_up:")
    for R_up in (1.45, 1.55, 1.70)
        p = EntrepreneurshipParams(; R_up)
        r = entrepreneurship_steady_state(p; verbosity = 0)
        @printf "  R_up = %.2f  →  entrepreneur share = %.3f,  top-10%% wealth = %.3f\n" R_up r.entre_share r.top10
    end
end
