######################################################################
# Rust (1987) engine replacement — partial-equilibrium steady state   #
######################################################################

# Costs are exogenous (no market to clear), so the "outer loop" is a single inner
# V/Λ fixed-point solve at the given env. The whole within-period problem — the
# keep/replace choice AND the regeneration — is library stages only; see `model.jl`.
#
# The replacement RATE is not an `at_end` moment: the `:decision` axis is transient
# and gone by block end, so the stationary `Λ` is a pure mileage distribution (over
# the post-reset boundary mileage `x⁻`). We reconstruct the per-mileage replacement
# probability `π(replace | x)` from the converged `(V, Λ)` in the driver (plain
# closures, no reaching into the block), the same softmax the logit stage applies:
#
#   v(x,keep)    = −c(x)        + β·EV(x),
#   v(x,replace) = −(RC + c(0)) + β·EV(0),
#   π(replace|x) = logistic( (v(x,replace) − v(x,keep)) / ε ),
#
# with `EV = V` the stationary boundary value. The manager acts on the mileage AFTER
# deterioration, so the choice-time distribution is `Λ` pushed through the
# deterioration matrix, `Λ_choice = Tᵀ·Λ`, and the aggregate replacement rate is
# `Σ_x Λ_choice(x)·π(replace|x)`.

include("model.jl")

using Printf

"""
Reconstruct the per-mileage replacement probability `π(replace | x)` and the aggregate replacement rate
from the converged stationary value `V` and mileage distribution `Λ`, using the SAME EV-logit softmax the
`LogitChoiceStage` applies internally. Pure driver-side arithmetic (no block internals): forms the two
choice-specific values via the deterioration expectation `T·V`, then a numerically stable two-option
softmax. Returns `(; π_replace, replacement_rate)`.
"""
function replacement_probabilities(p::RustParams, V::AbstractVector, Λ::AbstractVector)
    xgrid     = collect(range(0.0, p.x_max; length = p.N_x))
    T         = deterioration_matrix(p)
    v_keep    = -opcost.(xgrid, Ref(p)) .+ p.β .* V          # −c(x) + β·EV(x)
    v_replace = -(p.RC + opcost(0.0, p)) + p.β * V[1]        # −(RC+c(0)) + β·EV(0)   (scalar)
    π_replace = @. 1.0 / (1.0 + exp((v_keep - v_replace) / p.ε))  # logistic(Δ/ε), stable for these magnitudes
    Λ_choice  = T' * Λ                                       # mileage distribution at decision time (post-deterioration)
    return (; π_replace, Λ_choice,
              mean_operating_mileage = sum(Λ_choice .* xgrid),
              replacement_rate       = sum(Λ_choice .* π_replace))
end

"""
Solve the Rust engine-replacement steady state at the exogenous env and report the stationary mileage
mean, the aggregate replacement rate, and the mileage at which replacement becomes more likely than not.
Returns the stationary `(V, Λ)`, the moment, and the replacement diagnostics.
"""
function rust_steady_state(p = rust_params; verbosity = 1)
    hh  = rust_household(p)
    env = rust_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)

    V = vec(res.V)                                           # (mileage, decision=1) → mileage
    Λ = vec(res.Λ)
    rp = replacement_probabilities(p, V, Λ)
    xgrid = collect(range(0.0, p.x_max; length = p.N_x))
    i_half = findfirst(>=(0.5), rp.π_replace)                # first mileage with π(replace) ≥ 1/2
    x_half = i_half === nothing ? NaN : xgrid[i_half]

    if verbosity > 0
        @printf "Rust (1987) engine replacement (β = %.3f, ε = %.2f, RC = %.1f, c1 = %.2f)\n" p.β p.ε p.RC p.c1
        @printf "  mass(Λ)                 = %.6f\n"  sum(Λ)
        @printf "  mean operating mileage  = %.4f  (grid top %.1f)\n"  rp.mean_operating_mileage  p.x_max
        @printf "  mean boundary mileage   = %.4f  (post-reset, ∫x dΛ)\n"  m.mean_mileage
        @printf "  replacement rate        = %.4f per period\n"  rp.replacement_rate
        @printf "  π(replace) crosses ½ at mileage = %.2f\n"  x_half
        @printf "  π(replace) at x=0 / mid / top   = %.4f / %.4f / %.4f\n" rp.π_replace[1] rp.π_replace[(p.N_x+1)÷2] rp.π_replace[end]
        @printf "  V range = [%.2f, %.2f]\n"  minimum(V)  maximum(V)
        @printf "  VFI iters = %d, Λ iters = %d\n" res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_operating_mileage = rp.mean_operating_mileage,
              mean_mileage = m.mean_mileage, replacement_rate = rp.replacement_rate,
              π_replace = rp.π_replace, x_half, history = res.history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Rust (1987) engine-replacement steady state…")
    @time rust_steady_state()
end
