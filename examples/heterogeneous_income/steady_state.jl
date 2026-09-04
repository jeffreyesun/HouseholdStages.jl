###################################################################
# HIP steady state — single fixed-price solve, wealth by type      #
###################################################################

# Heterogeneous Income Profiles is a PARTIAL-equilibrium experiment: the
# return `r` and wage `w` are exogenous and fixed (no market to clear), so
# the solve is a SINGLE `solve_steady_state_given_env!` at the fixed env —
# no outer tatonnement loop. The only example-side work beyond the call is
# the per-type wealth decomposition, computed from the stationary `Λ` and
# the cell grid (plain aggregation, not a new stage).

include("model.jl")

using Printf

"""
Solve the HIP household at the fixed env and report the stationary wealth
distribution split by permanent earnings type θ. Mass is conserved within
each type (no Markov on `:income_type`); from a uniform start each type
carries an equal share, so per-type mean wealth is `(∫_type wealth dΛ) /
(mass in type)`. Returns `V`, `Λ`, the aggregate `A_mean`, and the per-type
mean-wealth and mass vectors.
"""
function hip_steady_state(p = hip_params; verbosity = 1)
    hh  = hip_household(p)
    env = hip_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res

    # Per-type wealth decomposition from the stationary Λ #
    #----------------------------------------------------#
    # Λ is shaped (N_w, n_ε, n_θ); the :income_type axis is position 3.
    out_layout = end_layout(hh)
    cells = cell_array(out_layout)
    nθ = length(p.θ_grid)

    type_mass        = [sum(Λ[:, :, k])                                       for k in 1:nθ]
    type_wealth_sum  = [sum(getproperty.(cells[:, :, k], :wealth) .* Λ[:, :, k]) for k in 1:nθ]
    type_mean_wealth = type_wealth_sum ./ type_mass

    A_mean = compute_moments(hh, Λ, env).A_mean

    if verbosity > 0
        @printf "HIP steady state (σ = %.1f, r = %.3f, w = %.2f, β = %.3f)\n" p.σ p.r p.w p.β
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  ΣΛ                 = %.6f\n" sum(Λ)
        @printf "  V finite           = %s\n"  all(isfinite, V)
        @printf "  aggregate A_mean   = %.4f\n" A_mean
        for k in 1:nθ
            @printf "  type θ = %.2f : mass = %.4f, mean wealth = %.4f\n" p.θ_grid[k] type_mass[k] type_mean_wealth[k]
        end
        @printf "  high/low wealth ratio = %.3f\n" type_mean_wealth[end] / type_mean_wealth[1]
    end

    return (; V, Λ, A_mean, type_mass, type_mean_wealth, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Heterogeneous Income Profiles steady state…")
    @time res = hip_steady_state()
    println(sum(res.Λ) ≈ 1 ? "Stationary Λ sums to 1." : "WARNING: ΣΛ = $(sum(res.Λ))")
end
