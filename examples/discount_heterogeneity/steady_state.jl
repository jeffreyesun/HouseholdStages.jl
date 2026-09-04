###################################################################
# Discount-heterogeneity steady state — product solve, wealth by β #
###################################################################

# A SINGLE fixed-price solve on the β-product household: `r` and `w` are
# exogenous, so there is no outer tatonnement. The product is block-diagonal
# across β-types, so the STANDARD `solve_steady_state_given_env!` runs
# directly on the moment-attached ChainStage — VFI to a fixed point and Λ to
# stationarity on the fused `(N_w, n_ε, n_β)` tensor, each β-slice
# converging independently (no cross-slice threading, unlike life-cycle).
# The only example-side work is the per-β wealth decomposition from `Λ`.

include("model.jl")

using Printf

"""
Solve the β-product household at the fixed env and report the stationary
wealth distribution split by discount-factor type. Each β-type is an
independent slice along the `:beta` axis (position 3 of `Λ`); mass is
conserved within a type. Returns `V`, `Λ`, the aggregate `A_mean`, and the
per-β mean-wealth and mass vectors. Verifies the central comparative-static:
more patient types hold more wealth.
"""
function discount_het_steady_state(p = discount_het_params; verbosity = 1)
    hh  = discount_het_household(p)
    env = discount_het_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res

    # Per-β wealth decomposition from the stationary Λ #
    #-------------------------------------------------#
    out_layout = end_layout(hh)
    cells = cell_array(out_layout)
    nβ = length(p.β_grid)

    β_mass        = [sum(Λ[:, :, k])                                          for k in 1:nβ]
    β_wealth_sum  = [sum(getproperty.(cells[:, :, k], :wealth) .* Λ[:, :, k]) for k in 1:nβ]
    β_mean_wealth = β_wealth_sum ./ β_mass

    A_mean = compute_moments(hh, Λ, env).A_mean

    if verbosity > 0
        @printf "Discount-heterogeneity steady state (σ = %.1f, r = %.3f, w = %.2f)\n" p.σ p.r p.w
        @printf "  β-types            = %s\n" string(p.β_grid)
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  ΣΛ                 = %.6f\n" sum(Λ)
        @printf "  V finite           = %s\n"  all(isfinite, V)
        @printf "  aggregate A_mean   = %.4f\n" A_mean
        for k in 1:nβ
            @printf "  β = %.3f : mass = %.4f, mean wealth = %.4f\n" p.β_grid[k] β_mass[k] β_mean_wealth[k]
        end
        monotone = all(diff(β_mean_wealth) .> 0)
        @printf "  wealth increasing in β : %s (patient hold more)\n" monotone
    end

    return (; V, Λ, A_mean, β_mass, β_mean_wealth, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving discount-heterogeneity steady state…")
    @time res = discount_het_steady_state()
    println(sum(res.Λ) ≈ 1 ? "Stationary Λ sums to 1." : "WARNING: ΣΛ = $(sum(res.Λ))")
end
