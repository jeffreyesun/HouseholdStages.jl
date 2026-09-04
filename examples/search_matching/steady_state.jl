#####################################################################
# Search & Matching — steady state (partial eq. + free-entry GE θ) #
#####################################################################

# Two drivers, both rolling their own outer loop (the library supplies only the
# per-env V/Λ fixed-point solve, per the "close-the-model loops live with the
# consumer" principle in `../PLAN.md`):
#
#   1. Partial equilibrium — fix tightness θ exogenously, one inner solve. Exercises
#      the stage's `FromEnv(:θ)` contract directly. Robust, no outer fixed point.
#   2. Free-entry GE — close θ with the firm's vacancy-posting condition
#      `κ = q(θ)·J` by bisection on the scalar residual `free_entry_residual(θ)`.
#      The firm value `J` is a closed-form geometric sum (model.jl), so the GE loop
#      is a scalar root-find on θ wrapped around the household solve — no firm
#      Bellman iteration, no bespoke household logic.

include("model.jl")

using Printf


# 1. Partial-equilibrium steady state at exogenous θ #
#----------------------------------------------------#

"""
Solve the household block at a fixed tightness `θ` (partial equilibrium). One inner
V/Λ fixed point via `solve_steady_state_given_env!`; returns moments and the seated
job-finding policy `p*(wealth)` on the unemployed slice, read from the `Matching`
`MixingStage` leaf.
"""
function search_matching_pe(p = search_matching_params; θ = 1.0, verbosity = 1)
    hh  = search_matching_household(p)
    env = (; θ, p.r, p.w, p.b_u)
    res = solve_steady_state_given_env!(hh, env)
    m   = res.moments

    p_find = HouseholdStages.policy(hh.buffer.stages[2])[:, 1]   # Matching leaf, unemployed slice
    p_find_mean = sum(p_find) / length(p_find)

    verbosity > 0 && @printf("  PE  θ = %.3f → employment = %.4f, K = %.4f, mean p* = %.4f (VFI %d, Λ %d)\n",
                             θ, m.employment, m.K_supplied, p_find_mean,
                             res.history.vfi_iters, res.history.lambda_iters)

    return (; θ, employment = m.employment, K = m.K_supplied, p_find, p_find_mean,
              V = res.V, Λ = res.Λ, q = vacancy_fill(θ, p))
end


# 2. Free-entry GE — bisection on θ #
#-----------------------------------#

"""
Close market tightness `θ` by the firm free-entry condition `κ = q(θ)·J`. Bisection
on `free_entry_residual(θ)` (model.jl) brackets the GE tightness; at the root the
household block is re-solved to report equilibrium employment and wealth. The firm
side is closed-form scalar arithmetic, so each bisection step costs only the firm
residual; the household block is solved once at the converged θ.
"""
function search_matching_ge(p = search_matching_params;
                            θ_lo = 0.05, θ_hi = 20.0,
                            tol = 1e-6, max_iter = 80, verbosity = 1)
    f_lo = free_entry_residual(θ_lo, p)
    f_hi = free_entry_residual(θ_hi, p)
    @assert f_lo * f_hi <= 0 "free-entry residual does not bracket a root on " *
        "[$θ_lo, $θ_hi]; got f($θ_lo)=$f_lo, f($θ_hi)=$f_hi. Adjust κ/z or the bracket."

    a, b = θ_lo, θ_hi
    fa = f_lo
    θ_star = 0.5 * (a + b)
    for it in 1:max_iter
        θ_star = 0.5 * (a + b)
        fm = free_entry_residual(θ_star, p)
        if abs(fm) <= tol || (b - a) / 2 <= tol
            verbosity > 0 && @printf("  GE  bisection converged: θ* = %.5f, residual = %.2e (iter %d)\n",
                                     θ_star, fm, it)
            break
        end
        if fa * fm < 0
            b = θ_star
        else
            a, fa = θ_star, fm
        end
    end

    pe = search_matching_pe(p; θ = θ_star, verbosity = 0)
    return (; θ = θ_star, residual = free_entry_residual(θ_star, p),
              employment = pe.employment, K = pe.K, q = pe.q,
              J = filled_job_value(p), V = pe.V, Λ = pe.Λ)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p = search_matching_params

    println("Search & matching — partial equilibrium across exogenous θ:")
    for θ in (0.5, 1.0, 2.0, 4.0)
        search_matching_pe(p; θ)
    end

    println("\nSearch & matching — free-entry general equilibrium:")
    @time ge = search_matching_ge(p)
    @printf("  θ*           = %.5f\n", ge.θ)
    @printf("  q(θ*)        = %.5f\n", ge.q)
    @printf("  J (job val)  = %.5f\n", ge.J)
    @printf("  free-entry   = %.2e  (κ = %.3f)\n", ge.residual, p.κ)
    @printf("  employment   = %.5f\n", ge.employment)
    @printf("  K_supplied   = %.5f\n", ge.K)
    @printf("  ΣΛ           = %.6f\n", sum(ge.Λ))
end
