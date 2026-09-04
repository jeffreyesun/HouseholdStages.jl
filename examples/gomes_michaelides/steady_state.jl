####################################################################
# Gomes–Michaelides steady state — participation by type and wealth #
####################################################################

# A SINGLE fixed-price solve on the `:ptype`-product household: returns, wage,
# and the participation cost are exogenous, so there is no outer tatonnement.
# The product is block-diagonal across preference types, so the STANDARD
# `solve_steady_state_given_env!` runs directly on the moment-attached
# ChainStage, each type's slice converging independently. All example-side work
# is reporting: pulling each type's seated risky-share policy `θ*(x)` from its
# `GaussianLoadingStage` leaf and computing participation rates (the fraction of
# mass with `θ* > 0`) per type and across the wealth distribution.

include("model.jl")

using Printf

"""
The per-type seated risky-share policy `θ*(x)`. Navigates the solved product:
the moment-wrapped ChainStage holds the `ProductStage` (stage 1); its bundled
components are the per-type Merton blocks; each block's `GaussianLoadingStage` leaf
is its LAST stage, whose `policy` is the seated `θ*(x)` on `(wealth, income)`.
"""
function gm_share_policies(hh, nptype)
    prod   = hh.buffer.stages[1]                                   # the ProductStage
    blocks = prod.buffer.components                                # per-type Merton blocks
    return [HouseholdStages.policy(blocks[i].buffer.stages[end])[:, :, 1] for i in 1:nptype]
end

"""
Solve the Gomes–Michaelides `:ptype`-product household at the fixed env and
report the participation margin: per-type participation rates and conditional
shares, plus participation across three equal-mass wealth bins (the central
comparative static — participation RISES with wealth).
"""
function gm_steady_state(p = gm_params; verbosity = 1)
    hh  = gm_household(p)
    env = gm_env(p)

    res = solve_steady_state_given_env!(hh, env)
    (; V, Λ, history) = res

    nθ      = length(p.σ_grid)
    θ       = gm_share_policies(hh, nθ)                            # θ[i] :: (N_w, n_income)
    mean_w  = compute_moments(hh, Λ, env).mean_wealth

    # Per-type mass, participation rate, and conditional mean share #
    #--------------------------------------------------------------#
    type_mass = [sum(Λ[:, :, i])                                            for i in 1:nθ]
    part_mass = [sum(Λ[:, :, i] .* (θ[i] .> 0))                             for i in 1:nθ]
    part_rate = part_mass ./ type_mass
    cond_share = [let m = Λ[:, :, i] .* (θ[i] .> 0)
                      sum(m) > 0 ? sum(θ[i] .* m) / sum(m) : 0.0
                  end for i in 1:nθ]

    # Participation across the wealth distribution (pooled over types/income) #
    #------------------------------------------------------------------------#
    # Marginal-over-wealth mass and participating mass at each wealth grid point.
    mass_w = [sum(Λ[j, :, :])                                               for j in 1:p.N_w]
    part_w = [sum(sum(Λ[j, :, i] .* (θ[i][j, :] .> 0)) for i in 1:nθ)       for j in 1:p.N_w]

    # Three equal-mass wealth bins (terciles of the cross-section by wealth).
    cum   = cumsum(mass_w) ./ sum(mass_w)
    edges = (searchsortedfirst(cum, 1 / 3), searchsortedfirst(cum, 2 / 3))
    bins  = (1:edges[1], (edges[1] + 1):edges[2], (edges[2] + 1):p.N_w)
    bin_part = [sum(part_w[b]) / sum(mass_w[b]) for b in bins]

    if verbosity > 0
        prem = sum(p.p_risky .* p.R_risky) - p.R_f
        @printf "Gomes–Michaelides steady state (premium = %.3f, F = %.3f utils)\n" prem p.F
        @printf "  σ-types            = %s\n" string(p.σ_grid)
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  ΣΛ                 = %.6f  (V finite: %s)\n" sum(Λ) all(isfinite, V)
        @printf "  aggregate mean wealth = %.4f\n" mean_w
        println("  — participation by preference type —")
        for i in 1:nθ
            @printf "  σ = %.2f : mass = %.3f, participation = %.3f, conditional θ̄ = %.3f\n" p.σ_grid[i] type_mass[i] part_rate[i] cond_share[i]
        end
        @printf "  overall participation rate = %.3f\n" (sum(part_mass) / sum(type_mass))
        println("  — participation across wealth (equal-mass terciles) —")
        @printf "  bottom third : %.3f\n" bin_part[1]
        @printf "  middle third : %.3f\n" bin_part[2]
        @printf "  top    third : %.3f\n" bin_part[3]
        @printf "  rises with wealth : %s\n" (bin_part[1] <= bin_part[2] <= bin_part[3])
    end

    return (; V, Λ, mean_wealth = mean_w, θ, part_rate, cond_share, bin_part, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Gomes–Michaelides participation steady state…")
    @time res = gm_steady_state()
    println(sum(res.Λ) ≈ 1 ? "Stationary Λ sums to 1." : "WARNING: ΣΛ = $(sum(res.Λ))")
end
