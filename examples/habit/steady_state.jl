###############################################
# Habit / rational addiction — steady state    #
###############################################

# Partial equilibrium (exogenous r): a single inner V/Λ solve. The point is the household block —
# a SINGLE savings choice that sets liquid wealth AND drives the addiction stock — built from
# existing stages via the auxiliary-choice-axis pattern (see model.jl). No market to clear.

include("model.jl")

using Printf

function habit_steady_state(p = HabitParams(); verbosity = 1)
    hh  = habit_household(p)
    res = solve_steady_state_given_env!(hh, NamedTuple())
    m   = compute_moments(hh, res.Λ, NamedTuple())
    if verbosity > 0
        @printf "Habit / rational addiction steady state (β=%.2f, α=%.3f, δ_S=%.2f)\n" p.β p.α p.δ_S
        @printf "  mass(Λ)     = %.6f\n" sum(res.Λ)
        @printf "  mean wealth = %.4f\n" m.mean_wealth
        @printf "  mean habit  = %.4f\n" m.mean_habit
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, m.mean_wealth, m.mean_habit, history = res.history)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving habit / rational-addiction steady state…")
    @time habit_steady_state()
end
