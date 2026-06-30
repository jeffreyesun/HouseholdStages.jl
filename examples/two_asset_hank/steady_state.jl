###############################################
# Two-asset HANK — steady state                #
###############################################

# Partial equilibrium (exogenous r_b, r_a): a single inner V/Λ solve. The household block — a single
# illiquid choice that sets the illiquid axis AND debits liquid by the deposit — is built from
# existing stages via the auxiliary-choice-axis pattern (see model.jl).

include("model.jl")

using Printf

function two_asset_steady_state(p = TwoAssetParams(); verbosity = 1)
    hh  = two_asset_household(p)
    res = solve_steady_state_given_env!(hh, NamedTuple())
    m   = compute_moments(hh, res.Λ, NamedTuple())
    if verbosity > 0
        @printf "Two-asset HANK steady state (r_b=%.3f, r_a=%.3f, κ=%.3f)\n" p.r_b p.r_a p.κ
        @printf "  mass(Λ)        = %.6f\n" sum(res.Λ)
        @printf "  mean liquid    = %.4f\n" m.mean_liquid
        @printf "  mean illiquid  = %.4f\n" m.mean_illiquid
        @printf "  illiquid share = %.3f\n" (m.mean_illiquid / (m.mean_liquid + m.mean_illiquid))
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, m.mean_liquid, m.mean_illiquid, history = res.history)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving two-asset HANK steady state…")
    @time two_asset_steady_state()
end
