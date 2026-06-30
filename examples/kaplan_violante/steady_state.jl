###############################################
# Kaplan–Violante (2014) — steady state        #
###############################################
#
# Partial equilibrium (exogenous r_b, r_a): a single inner V/Λ solve. The block
# is the two-asset auxiliary-choice-axis pattern with a FIXED adjustment cost
# and an explicit no-adjust option — built from existing stages (see model.jl).

include("model.jl")

using Printf

function kv_steady_state(p = KVParams(); verbosity = 1)
    hh  = kv_household(p)
    res = solve_steady_state_given_env!(hh, NamedTuple())
    m   = compute_moments(hh, res.Λ, NamedTuple())
    if verbosity > 0
        @printf "Kaplan–Violante steady state (r_b=%.3f, r_a=%.3f, κ_f=%.3f)\n" p.r_b p.r_a p.κ_f
        @printf "  mass(Λ)         = %.6f\n" sum(res.Λ)
        @printf "  mean liquid     = %.4f\n" m.mean_liquid
        @printf "  mean illiquid   = %.4f\n" m.mean_illiquid
        @printf "  illiquid share  = %.3f\n" (m.mean_illiquid / (m.mean_liquid + m.mean_illiquid))
        @printf "  wealthy-HtM share = %.3f\n" m.frac_htm
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, m.mean_liquid, m.mean_illiquid, m.frac_htm, history = res.history)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Kaplan–Violante steady state…")
    @time kv_steady_state()
end
