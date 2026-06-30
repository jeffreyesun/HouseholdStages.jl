##################################################################
# Endogenous separations — stationary steady state (partial eq.) #
##################################################################

# Tightness `θ`, wage scale `w` and benefit `b_u` are exogenous (partial equilibrium):
# no market to clear, so the "outer loop" is a single `solve_steady_state_given_env!`.
# The driver reports mass(Λ) ≈ 1, the employment rate, the TOTAL separation rate
# (exogenous δ + endogenous quits), the quit threshold in `z`, and the mean `z` among
# the employed vs. the unconditional mean — and checks that quits occur at low `z`.

include("model.jl")

using Printf


# Policy extraction (reporting only — outside the block) #
#-------------------------------------------------------#

"""
The seated keep/quit policy of a solved household: the `(N_z, 2)` array of chosen
`:emp` indices (1 = unemployed, 2 = employed) per `(z, origin-emp)` cell. The unique
policy-bearing leaf is the `QuitChoice` `ArgmaxStage` (`SearchMatchingStage` exposes
no `policy` method), so we pluck it by the `policy` method rather than a positional
index.
"""
function quit_policy(hh)
    leaves = filter(s -> !(s isa HouseholdStages.ChainStage) &&
                         hasmethod(HouseholdStages.policy, Tuple{typeof(s)}),
                    collect(hh.buffer.stages))
    return HouseholdStages.policy(only(leaves))
end


# Stationary solve #
#------------------#

"""
Solve the endogenous-separations household at the exogenous `(w, b_u)` env (partial
equilibrium). One inner V/Λ fixed point via `solve_steady_state_given_env!`. Reports
the total separation rate and quit threshold by combining the seated `QuitChoice`
policy with the stationary distribution and the exogenous rate `δ`.
"""
function endogenous_separations_steady_state(p = endogenous_separations_params; verbosity = 1)
    hh  = endogenous_separations_household(p)
    env = endogenous_separations_env(p)
    res = solve_steady_state_given_env!(hh, env)
    m   = res.moments

    _, z = z_process(p)

    employment = m.employment
    mean_z_emp = employment > 0 ? m.z_emp / employment : NaN
    mean_z     = m.z_uncond                       # ΣΛ = 1, so this is the unconditional mean

    # Quit margin. The employed-origin column (index 2) of the QuitChoice policy gives
    # quit (dest 1) vs. stay (dest 2) per z. Λ is the START-of-period distribution, so
    # the employed mass facing the quit decision is Λ[:, emp].
    pol       = quit_policy(hh)
    Λ         = reshape(res.Λ, p.N_z, 2)
    quit_z    = pol[:, 2] .== 1                    # employed who choose to quit, by z
    Λ_emp     = Λ[:, 2]
    emp_mass  = sum(Λ_emp)
    quit_rate = emp_mass > 0 ? sum(Λ_emp[quit_z]) / emp_mass : 0.0
    total_sep = quit_rate + (1 - quit_rate) * p.δ  # quit now, else exogenous δ later

    # Quit threshold: the highest z still quit (boundary). Inf-symbol if no quits.
    quit_idx       = findall(quit_z)
    quit_threshold = isempty(quit_idx) ? -Inf : z[maximum(quit_idx)]

    if verbosity > 0
        @printf("  ΣΛ              = %.6f\n", sum(res.Λ))
        @printf("  employment      = %.4f\n", employment)
        @printf("  exogenous δ     = %.4f\n", p.δ)
        @printf("  endog quit rate = %.4f\n", quit_rate)
        @printf("  TOTAL sep rate  = %.4f\n", total_sep)
        @printf("  quit threshold z= %s\n", isfinite(quit_threshold) ? @sprintf("%.4f", quit_threshold) : "none (no quits)")
        @printf("  mean z | emp    = %.4f\n", mean_z_emp)
        @printf("  mean z (uncond) = %.4f\n", mean_z)
        @printf("  z grid          = %s\n", string(round.(z; digits = 3)))
        @printf("  quit-by-z (emp) = %s\n", string(Int.(quit_z)))
        @printf("  VFI iters %d, Λ iters %d\n", res.history.vfi_iters, res.history.lambda_iters)
    end

    return (; employment, quit_rate, total_sep, quit_threshold, mean_z_emp, mean_z,
              quit_z, z, V = res.V, Λ = res.Λ, hh)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Endogenous separations — stationary steady state:")
    @time res = endogenous_separations_steady_state()

    # Quits must occur at LOW z: the quit set is a low-z prefix (monotone — once a
    # worker stays at some z, they stay at every higher z).
    stay = .!res.quit_z
    monotone = all(diff(Int.(stay)) .>= 0)        # quit (0) then stay (1), non-decreasing
    @printf("  quits at low z (monotone): %s\n", monotone)

    @assert monotone "quit policy is not a low-z prefix"
    @assert res.mean_z_emp >= res.mean_z - 1e-8 "employed should be positively selected on z"
    @assert abs(sum(res.Λ) - 1) < 1e-6 "stationary mass leaked (ΣΛ ≠ 1)"
    @assert all(isfinite, res.V) "value function has non-finite entries"
    println("  ✓ all checks passed")
end
