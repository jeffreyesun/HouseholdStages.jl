###############################################
# Constantinides internal habit — steady state #
###############################################

# Partial equilibrium (exogenous r): a single inner V/Λ solve. The point is the household block —
# a SINGLE savings choice that sets liquid wealth AND drives the internal habit stock — built from
# existing stages via the auxiliary-choice-axis pattern (see model.jl). Only the felicity differs
# from `examples/habit`: CRRA over the Constantinides surplus `c − γS`. No market to clear.

include("model.jl")

using Printf

"""
Verify the consumption-floor regularization is innocuous: over the stationary distribution, return the
minimum realised surplus `c − γS` and the mass share at or below the floor `c_floor`. The chosen
consumption is reconstructed from the `Choose` savings policy and the distribution entering the choice
(income shock + receipt applied to the stationary law). A minimum surplus well above `c_floor` and a
floor mass of essentially zero confirm the floored (strict-`−∞`) region carries no equilibrium mass.
"""
function surplus_diagnostic(p, hh, Λ_ss)
    wgrid = collect(range(0.0, p.w_max; length = p.N_w))
    Sgrid = collect(range(0.0, p.S_max; length = p.N_S))
    Λ1  = copy(forward!(hh.buffer.stages[1], copy(Λ_ss)))   # after IncomeShock
    Λ2  = copy(forward!(hh.buffer.stages[2], copy(Λ1)))     # after Receipt → distribution entering Choose
    pol = HouseholdStages.policy(hh.buffer.stages[3])        # chosen savings index per (x, S, y) cell
    minsurp, floormass, tot = Inf, 0.0, sum(Λ2)
    for I in CartesianIndices(Λ2)
        Λ2[I] > 1e-12 || continue
        surplus = (wgrid[I[1]] - wgrid[pol[I]]) - p.γ * Sgrid[I[2]]   # c − γS, c = x − b'
        minsurp = min(minsurp, surplus)
        surplus <= p.c_floor && (floormass += Λ2[I])
    end
    return (; minsurp, floor_mass_share = floormass / tot)
end

function constantinides_habit_steady_state(p = ConstantinidesHabitParams(); verbosity = 1)
    hh  = constantinides_habit_household(p)
    res = solve_steady_state_given_env!(hh, NamedTuple())
    m   = compute_moments(hh, res.Λ, NamedTuple())
    d   = surplus_diagnostic(p, hh, res.Λ)
    if verbosity > 0
        @printf "Constantinides internal-habit steady state (β=%.2f, σ=%.1f, γ=%.2f, δ_S=%.2f)\n" p.β p.σ p.γ p.δ_S
        @printf "  mass(Λ)     = %.6f\n" sum(res.Λ)
        @printf "  mean wealth = %.4f\n" m.mean_wealth
        @printf "  mean habit  = %.4f\n" m.mean_habit
        @printf "  min surplus = %.4f  (floor c_floor = %.1e; mass at floor = %.2e)\n" d.minsurp p.c_floor d.floor_mass_share
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, m.mean_wealth, m.mean_habit,
              min_surplus = d.minsurp, floor_mass_share = d.floor_mass_share, history = res.history)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Constantinides internal-habit steady state…")
    @time constantinides_habit_steady_state()
end
