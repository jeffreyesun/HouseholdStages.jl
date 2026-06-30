############################################################
# Borrowing-Limit Steady State — natural vs ad-hoc floors   #
############################################################

# Fixed-`r` partial-equilibrium solves (no market clearing): the SAME spine is
# solved at two asset-grid floors — a near-natural floor (the deepest sustainable
# debt) and a tighter ad-hoc floor — and the stationary wealth distributions are
# compared. The per-env inner work (V backward to a fixed point, Λ forward to
# stationarity) is delegated to `HouseholdStages.solve_steady_state_given_env!`.

include("model.jl")

using Printf

"""
Solve the borrowing-limit steady state at the fixed return `r` for a given asset
floor `a_min` (single inner V/Λ solve) and return the aggregate position
`A_mean`, the borrowing share, and the constrained mass.
"""
function borrowing_constraint_steady_state(a_min::Real, p = borrowing_constraint_params; verbosity = 1)
    hh  = borrowing_constraint_household(a_min, p)
    env = borrowing_constraint_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    if verbosity > 0
        @printf "  a_min = %+.3f : ΣΛ = %.6f, A_mean = %+.4f, frac borrowing = %.4f, frac at floor = %.4f, VFI = %d\n" a_min sum(res.Λ) m.A_mean m.frac_borrowing m.frac_constrained res.history.vfi_iters
    end

    return (; a_min, V = res.V, Λ = res.Λ,
              A_mean = m.A_mean, frac_borrowing = m.frac_borrowing,
              frac_constrained = m.frac_constrained, history = res.history)
end

"""
Compare the natural-limit economy (floor just inside `−y_min/r`) against a
tighter ad-hoc floor. Both solve the identical spine; only the grid floor differs.
"""
function compare_limits(p = borrowing_constraint_params;
                        natural_floor = natural_limit(p) + 0.33,   # just inside −y_min/r ≈ −3.33
                        adhoc_floor   = -1.0,
                        verbosity     = 1)
    verbosity > 0 && @printf "Borrowing limits (β = %.2f, σ = %.1f, r = %.4f); natural limit −y_min/r = %.3f\n" p.β p.σ p.r natural_limit(p)
    verbosity > 0 && println("  NATURAL-limit economy (floor near −y_min/r):")
    nat = borrowing_constraint_steady_state(natural_floor, p; verbosity)
    verbosity > 0 && println("  AD-HOC-limit economy (tighter exogenous floor):")
    adh = borrowing_constraint_steady_state(adhoc_floor, p; verbosity)
    return (; natural = nat, adhoc = adh)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving borrowing-limit steady states (natural vs ad-hoc)…")
    @time res = compare_limits()
    @printf "ΣΛ natural = %.6f, ΣΛ ad-hoc = %.6f\n" sum(res.natural.Λ) sum(res.adhoc.Λ)
end
