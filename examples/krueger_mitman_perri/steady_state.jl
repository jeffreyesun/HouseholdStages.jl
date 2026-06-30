###############################################################
# KMP Steady State — GE tatonnement on aggregate capital K     #
###############################################################

# Damped tatonnement on aggregate capital K (the Aiyagari outer loop). At each
# pass the per-K inner work (V backward to a fixed point + Λ forward to
# stationarity) is delegated to `HouseholdStages.solve_steady_state_given_env!`;
# the outer K loop is rolled here. The UI policy (replacement rate ρ, balanced-
# budget tax τ) is fixed by the calibration and enters through `kmp_env`, so the
# only object iterated is K.

include("model.jl")

using Printf

"""
Solve the Krueger–Mitman–Perri GE steady state by damped tatonnement on
aggregate capital K: at each pass solve the inner V/Λ problem at the current K
(prices `r(K), w(K)` and the fixed UI policy) and nudge
`K ← K + update_speed·(K_supplied − K)`. Hard-argmax savings makes `K_supplied`
a step function in K, so the residual has a discretization floor; `rtol = 0.02`
accepts it (the Aiyagari convention).
"""
function kmp_steady_state(p = kmp_params;
                          K_init       = 6.0,
                          update_speed = 0.05,
                          rtol         = 2e-2,
                          max_iter     = 500,
                          verbosity    = 1)
    hh = kmp_household(p)
    lm = kmp_labor_market(p)

    verbosity > 0 && @printf "Labor market: π_u = %.4f, π_e = %.4f, L = %.4f, balanced-budget τ = %.4f\n" lm.π_u lm.π_e lm.L lm.τ

    K = K_init
    iterations = 0
    K_err = Inf
    residual_history = Float64[]
    V, Λ = nothing, nothing

    while iterations < max_iter
        env = kmp_env(K, p)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        (; V, Λ, history) = res; (; vfi_iters, lambda_iters) = history

        K_supplied = compute_moments(hh, Λ, env).K_supplied
        K_err = abs(K_supplied - K) / K
        push!(residual_history, K_err)
        iterations += 1

        verbosity > 0 && @printf "  iter %d: K = %.4f → K_supplied = %.4f, K_err = %.6f, r = %.4f, VFI = %d, Λ = %d\n" iterations K K_supplied K_err env.r vfi_iters lambda_iters

        K_err <= rtol && break
        K += update_speed * (K_supplied - K)
    end

    converged = K_err <= rtol
    converged || @warn "KMP tatonnement stuck at tolerance $rtol after $iterations iterations."

    env = kmp_env(K, p)
    (; r, w) = env
    return (; K, r, w, τ = lm.τ, ρ = p.ρ, L = lm.L, V, Λ,
              iterations, converged, residual_history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Krueger–Mitman–Perri GE steady state…")
    @time res = kmp_steady_state()
    println(res.converged ? "Converged in $(res.iterations) outer iterations." :
                            "DID NOT CONVERGE in $(res.iterations) outer iterations.")
    @printf "  K   = %.4f\n" res.K
    @printf "  r   = %.4f   (1/β − 1 = %.4f)\n" res.r (1 / kmp_params.β - 1)
    @printf "  w   = %.4f\n" res.w
    @printf "  τ   = %.4f   (balanced-budget UI tax)\n" res.τ
    @printf "  ΣΛ  = %.6f\n" sum(res.Λ)
end
