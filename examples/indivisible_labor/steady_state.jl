###################################################################
# Indivisible labor — partial-equilibrium stationary distribution  #
###################################################################

# Single fixed-(r, w) solve: the household block backward-iterates V to its
# fixed point and forward-iterates Λ to stationarity at the prices in
# `indivisible_labor_env`, delegated to `solve_steady_state_given_env!`. No
# market clearing — the extensive-margin participation choice is the object of
# interest, and a fixed wage makes its response to the productivity state and to
# wealth transparent.

include("model.jl")

using Printf

"""
Solve the indivisible-labor stationary distribution at fixed `(r, w)` and report the
aggregate moments: participation rate `∫ 1{work} dΛ / mass`, mean hours `n̄ ·
participation rate`, and mean wealth `∫ wealth dΛ / mass`. Also returns the
participation rate split by productivity state ε, to show the extensive margin
responding to the wage.
"""
function indivisible_labor_steady_state(p = indivisible_labor_params; verbosity = 1)
    hh  = indivisible_labor_household(p)
    env = indivisible_labor_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res
    m    = compute_moments(hh, Λ, env)
    mass = sum(Λ)

    participation = m.participation_agg / mass
    mean_hours    = p.nbar * participation
    mean_wealth   = m.mean_wealth_agg / mass

    # Participation rate by productivity state ε: marginalize Λ over wealth and the
    # participation indicator, weighting by the indicator. (Λ axes: wealth, income, participation.)
    pcoord = p.part_grid                                   # participation values aligned to axis 3
    part_by_eps = map(enumerate(p.y_grid)) do (j, ε)
        slab = @view Λ[:, j, :]
        work_mass  = sum(pcoord[k] * slab[i, k] for i in axes(slab, 1), k in axes(slab, 2))
        total_mass = sum(slab)
        (ε, total_mass > 0 ? work_mass / total_mass : 0.0)
    end

    # Participation rate by wealth tercile: the reservation-wealth (income) effect —
    # with CRRA the consumption value of labor income falls as wealth rises, so the
    # extensive margin should slope DOWN in wealth. Terciles of the wealth marginal.
    wmarg   = vec(sum(Λ, dims = (2, 3)))                  # mass by wealth grid point
    cumw    = cumsum(wmarg) ./ mass
    edges   = (findfirst(≥(1/3), cumw), findfirst(≥(2/3), cumw))
    work_by_wcell = [sum(pcoord[k] * Λ[i, j, k] for j in axes(Λ, 2), k in axes(Λ, 3))
                     for i in axes(Λ, 1)]
    function tercile_rate(lo, hi)
        wm = sum(@view wmarg[lo:hi]); wm > 0 ? sum(@view work_by_wcell[lo:hi]) / wm : 0.0
    end
    part_by_wealth = (low  = tercile_rate(1, edges[1]),
                      mid  = tercile_rate(edges[1] + 1, edges[2]),
                      high = tercile_rate(edges[2] + 1, length(wmarg)))

    if verbosity > 0
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
    end

    (; V, Λ, participation, mean_hours, mean_wealth, part_by_eps, part_by_wealth,
       env.r, env.w, mass, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p = indivisible_labor_params
    println("Solving indivisible-labor (Rogerson) stationary distribution…")
    @time res = indivisible_labor_steady_state(p)
    @printf "  r                 = %.4f\n" res.r
    @printf "  w                 = %.4f\n" res.w
    @printf "  participation rate = %.4f\n" res.participation
    @printf "  mean hours         = %.4f\n" res.mean_hours
    @printf "  mean wealth        = %.4f\n" res.mean_wealth
    @printf "  ΣΛ (mass)          = %.6f\n" res.mass
    @printf "  V finite           = %s\n" all(isfinite, res.V)
    println("  participation rate by productivity state ε:")
    for (ε, rate) in res.part_by_eps
        @printf "    ε = %.2f  →  P(work) = %.4f\n" ε rate
    end
    @printf "  participation by wealth tercile: low = %.4f, mid = %.4f, high = %.4f\n" res.part_by_wealth.low res.part_by_wealth.mid res.part_by_wealth.high
end
