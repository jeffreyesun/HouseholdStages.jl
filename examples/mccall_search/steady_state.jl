#############################################################
# McCall search — stationary steady state (partial equilib.) #
#############################################################

# Returns/benefits are exogenous (partial equilibrium): no market to clear, so the
# "outer loop" is a single `solve_steady_state_given_env!`. The driver reports the
# usual diagnostics (mass(Λ) ≈ 1, employment, mean accepted wage) and EXTRACTS the
# reservation wage from the seated `AcceptReject` `ArgmaxStage` policy — the lowest
# wage the unemployed accept — and checks the accept policy is monotone in the wage.

include("model.jl")

using Printf


# Policy extraction (reporting only — outside the block) #
#-------------------------------------------------------#

"""
The seated accept/reject policy of a solved McCall household: the `(N_w, 2)` array of
chosen `:emp` indices (1 = unemployed, 2 = employed) per `(wage, origin-emp)` cell.
The unique policy-bearing leaf is the `AcceptReject` `ArgmaxStage` (no savings stage
in this chain), so we pluck it by the `policy` method rather than a positional index.
"""
function accept_policy(hh)
    leaves = filter(s -> !(s isa HouseholdStages.ChainStage) &&
                         hasmethod(HouseholdStages.policy, Tuple{typeof(s)}),
                    collect(hh.buffer.stages))
    return HouseholdStages.policy(only(leaves))
end

"""
The reservation wage: the lowest wage at which an UNEMPLOYED origin (`:emp` column 1)
chooses to accept (destination index 2 = employed). Returns `Inf` if no offer is
accepted. Also returns the per-wage accept indicator for the monotonicity check.
"""
function reservation_wage(hh, p = mccall_params)
    grid   = wage_grid(p)
    pol    = accept_policy(hh)            # (N_w, 2): [:, 1] = unemployed origin
    accept = pol[:, 1] .== 2             # accept iff destination = employed
    idx    = findfirst(accept)
    w_star = isnothing(idx) ? Inf : grid[idx]
    return (; w_star, accept, grid)
end


# Stationary solve #
#------------------#

"""
Solve the McCall household at the exogenous benefit `b_u` (partial equilibrium). One
inner V/Λ fixed point via `solve_steady_state_given_env!`; returns moments plus the
reservation wage and accept indicator read off the seated `AcceptReject` policy.
"""
function mccall_steady_state(p = mccall_params; verbosity = 1)
    hh  = mccall_household(p)
    env = mccall_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = res.moments

    employment = m.employment
    mean_wage  = employment > 0 ? m.wage_emp / employment : NaN
    rw         = reservation_wage(hh, p)

    if verbosity > 0
        @printf("  ΣΛ            = %.6f\n", sum(res.Λ))
        @printf("  employment    = %.4f\n", employment)
        @printf("  mean acc wage = %.4f\n", mean_wage)
        @printf("  reservation w = %.4f\n", rw.w_star)
        @printf("  VFI iters %d, Λ iters %d\n", res.history.vfi_iters, res.history.lambda_iters)
    end

    return (; employment, mean_wage, w_star = rw.w_star, accept = rw.accept,
              V = res.V, Λ = res.Λ, hh)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("McCall reservation-wage search — stationary steady state:")
    @time res = mccall_steady_state()

    # Monotonicity: the accept indicator must be non-decreasing in the wage (reject
    # below the reservation wage, accept above it).
    monotone = all(diff(Int.(res.accept)) .>= 0)
    @printf("  accept policy monotone in wage: %s\n", monotone)

    @assert isfinite(res.w_star) "no offer is accepted — widen the wage grid or lower b_u"
    @assert monotone "accept policy is not monotone in the wage"
    @assert abs(sum(res.Λ) - 1) < 1e-6 "stationary mass leaked (ΣΛ ≠ 1)"
    @assert all(isfinite, res.V) "value function has non-finite entries"
    println("  ✓ all checks passed")
end
