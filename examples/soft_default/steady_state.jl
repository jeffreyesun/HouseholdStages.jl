####################################################################
# Soft-default steady state — partial equilibrium                   #
####################################################################

# The return/interest `r` is exogenous, so there is no market to clear: the
# "outer loop" is a single inner V/Λ fixed-point solve at the given env (contrast
# the Aiyagari tatonnement on K̄). The whole point is that the household block is
# library stages only — the soft-default choice is a `MixingStage` blending a
# clean-slate RESET kernel and the keep-the-balance-sheet IDENTITY at convex cost,
# which is MASS-CONSERVING (the defaulting mass resets to zero wealth and stays in
# the population). See `model.jl`.

include("model.jl")

using Printf

"""
Solve the soft-default household steady state at the exogenous env and report the
wealth/debt distribution and the seated default-probability policy `θ*(w)` (the
share routed through the clean-slate reset kernel `K_A`). Returns the stationary
`(V, Λ)`, the moments, the default policy, and the mass-weighted average default
probability.
"""
function soft_default_steady_state(p = soft_default_params; env = soft_default_env(p), verbosity = 1)
    hh  = soft_default_household(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    wealth_axis = GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :linear)
    w_grid      = collect(Float64, axisvalues(wealth_axis))

    θ = HouseholdStages.policy(hh.buffer.stages[2])        # DefaultChoice (MixingStage) leaf, θ(w, income)
    Λ = res.Λ
    avg_default = sum(θ .* Λ) / sum(Λ)                     # mass-weighted mean default probability

    # Default should concentrate in the debt region: compare the mass-weighted mean
    # θ on the debt slice (w < 0) vs the asset slice (w ≥ 0).
    debt_rows  = findall(<(0), w_grid)
    asset_rows = findall(>=(0), w_grid)
    mass_debt  = sum(@view Λ[debt_rows, :]);  mass_asset = sum(@view Λ[asset_rows, :])
    θ_debt  = mass_debt  > eps() ? sum(θ[debt_rows, :]  .* Λ[debt_rows, :])  / mass_debt  : 0.0
    θ_asset = mass_asset > eps() ? sum(θ[asset_rows, :] .* Λ[asset_rows, :]) / mass_asset : 0.0

    if verbosity > 0
        @printf "Soft-default steady state (r = %.3f, σ = %.1f, κ = %.2f, w∈[%.2f, %.2f])\n" env.r p.σ p.cost_curvature p.w_min p.w_max
        @printf "  mass(Λ)              = %.6f\n"      sum(Λ)
        @printf "  mean wealth          = %.4f\n"      m.mean_wealth
        @printf "  debt share (w<0)     = %.4f  (mean debt among debtors = %.4f)\n" m.debt_share (mass_debt > eps() ? m.mean_debt / m.debt_share : 0.0)
        @printf "  avg default prob θ̄   = %.4f\n"      avg_default
        @printf "  θ̄ | debt vs assets   = %.4f vs %.4f  (default concentrates in debt = %s)\n" θ_debt θ_asset (θ_debt > θ_asset)
        @printf "  θ*(w) range          = [%.4f, %.4f]  (interior = %s)\n" minimum(θ) maximum(θ) (maximum(θ) < 1.0 && maximum(θ) > 0.0)
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ, mean_wealth = m.mean_wealth, debt_share = m.debt_share,
              avg_default, θ_debt, θ_asset, default_policy = θ, w_grid, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving soft-default steady state…")
    @time soft_default_steady_state()
end
