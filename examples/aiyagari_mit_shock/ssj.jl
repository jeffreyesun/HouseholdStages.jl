##########################################################
# Aiyagari MIT Shock — Sequence-Space Derivatives Demo    #
##########################################################

# The household layer's env-derivative services at the Aiyagari steady state, on the 3-stage
# chain `IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`. The demo input is `env.r` and
# the output is the moment `K_supplied`: the household utility here is env-independent by
# construction, so a price shock reaches the chain only through `IncomeStage`'s budget.
#
# `expectation_vectors` — fake-news step 2, the `𝐊ᵀ` recursion — takes no env argument, so the
# kernels must have been seated by a prior `backward!` at `env_ss`; the script's re-solve below
# does that seating. `compute_fake_news_ssj` assembles the sequence-space Jacobian
# `J[t, s] = ∂K_supplied_t/∂r_s` in either of two modes: `:dual` sweeps one tangent-seated Dual
# chain and is exact at every anticipation distance, `:fd` sweeps two primal lanes at
# `env_ss ± h` and is `O(h²)`. `compute_steady_state_gradient` gives the permanent-shock
# comparative static `∂K_supplied/∂r`; it re-solves the steady state at `env_ss ± h` instead of
# differentiating it, and `:fd` is its only mode, because differentiating fixed points is out of
# scope for the package (`STAGES_ARCHITECTURE.md` §11, "What is differentiated, and what is not").
#
# THE SELF-CHECK, AND WHAT LIMITS IT. `max|J_dual - J_fd|` is the `:fd` lane's own error rather
# than a disagreement between two objects: `:dual` is exact, so the gap is what `h` costs.
#
# Both modes sweep from the SAME seated `(V_ss, Λ_ss)` buffers, so the steady-state solve's own
# error enters both lanes and largely cancels in the difference: the gap does not hit the
# `lambda_tol` wall an INDEPENDENTLY re-solved reference does. But those buffers are also the
# point being linearized about, so the solve tolerance changes the OBJECT rather than the check on
# it: `max|J|` is 12.506 at the package default `lambda_tol = 1e-6` against 12.550 from 1e-9 down,
# the default leaving Λ about 0.4% short. Which base point the drivers use is therefore not left to
# the caller — each solves for the steady state of the `env` it is handed, and the tâtonnement that
# finds `K_ss` runs its own `rtol = 2e-2` with no reason to solve the inner problems tightly. The
# explicit solve below is for `expectation_vectors`, which is shown on its own before any driver
# runs and needs the kernels seated. Linearized about a converged Λ, the gap is a clean function of
# `h` alone.
#
# At `vfi_tol = lambda_tol = 1e-11` (fresh steady state per point, `max|J| = 12.550`):
#
#     h        1e-3     1e-4     1e-5     1e-6     1e-7     1e-8
#     max|Δ|   3.9e-3   1.7e-4   8.4e-7   6.1e-9   5.8e-8   8.4e-7
#
# The first two points are inflated by argmax NODE SWITCHING: at a step that coarse the two `±h`
# lanes re-solve on opposite sides of a discrete-choice margin, and a cell that switches
# contributes a step rather than a derivative. Below the minimum at `h = 1e-6` the rise is
# roundoff, `O(ε_mach/h)`. The two arms are the `O(h²) + O(ε/h)` trade, and the minimum is where
# to sit.
#
# THE STEADY-STATE GRADIENT re-solves at `env ± h`, so its floor IS the tolerance those re-solves
# run at. It returns `∂K_supplied/∂r ≈ 1243.32`, stable to 1e-5 relative across warm starts and to
# 6e-4 relative across `h ∈ [1e-5, 1e-4]`. Run at the package solver defaults instead, the same
# call returns 660.12 from a cold warm start and 442.09 from a converged one — same call, same
# `h`, differing only in the buffers it inherited: noise of about the right magnitude, not a
# derivative. Which is why the driver runs those re-solves at `SS_PRECONDITION_TOL` and not at the
# solver's defaults.

include("transition.jl")          # pulls in model.jl and mit_steady_state

using Printf

# The FD step each service wants at this input. `h` is input-scaled: an input `K_supplied` barely
# responds to hits the `O(ε/h)` floor at a much larger `h`, and these values are `:r`'s.
const SSJ_H     = 1e-5
const SS_GRAD_H = 3e-5

"""
Run the household layer's sequence-space and steady-state derivative services at the pre-shock
Aiyagari steady state and print their headline entries.
"""
function ssj_demo(; T_horizon::Int = 30, p = mit_shock_params,
                    verbose::Bool = true)
    # 1. Pre-shock steady state (A = 1)
    verbose && println("Computing pre-shock steady state (A = 1)…")
    ss     = mit_steady_state(p; A = 1.0, verbosity = 0)
    hh     = ss.hh
    K_ss   = ss.K
    env_ss = (; K = K_ss, r = ss.r, w = ss.w)
    verbose && @printf "  K_ss = %.4f, r = %.4f, w = %.4f\n" K_ss ss.r ss.w

    # `expectation_vectors` below runs on its own, ahead of any driver, and reads kernels a prior
    # `backward!` seated — so the steady state is solved here at the tolerance a derivative wants.
    # The tâtonnement above stops at its own aggregate `rtol` and leaves its inner solves at the
    # package defaults. The drivers do this for themselves; this call is for the step that does not.
    ss_ss = solve_steady_state_given_env!(hh, env_ss;
                                          vfi_tol    = HouseholdStages.SS_PRECONDITION_TOL,
                                          lambda_tol = HouseholdStages.SS_PRECONDITION_TOL)
    V_ss, Λ_ss = ss_ss.V, ss_ss.Λ

    # 2. Expectation vectors for the K_supplied integrand
    verbose && println("\nRunning expectation_vectors over $T_horizon periods…")
    𝓔 = expectation_vectors(hh, cell -> cell.wealth, T_horizon)
    if verbose
        println("  produced $(length(𝓔)) expectation arrays of shape $(size(𝓔[1])).")
        println("  ⟨𝓔[t], Λ_ss⟩ for t = 0..min(5, T_horizon-1):")
        for t in 1:min(6, T_horizon)
            @printf "    t = %d: %.4f\n" (t-1) sum(𝓔[t] .* Λ_ss)
        end
    end

    # 3. Sequence-space Jacobian ∂K_supplied_t/∂r_s, both modes
    verbose && println("\nComputing the sequence-space Jacobian in both modes…")
    pair = (; inputs = (:r,), outputs = (:K_supplied,))
    J    = compute_fake_news_ssj(hh, env_ss, T_horizon; pair..., mode = :dual)
    J_fd = compute_fake_news_ssj(hh, env_ss, T_horizon; pair..., mode = :fd, h = SSJ_H)
    gap  = maximum(abs, J .- J_fd)

    if verbose
        @printf "  J : %s ; max|J| = %.4f\n" size(J) maximum(abs, J)
        @printf "  self-check  max|J_dual - J_fd| = %.3e  (%.2e relative) at h = %.0e\n" gap (gap / maximum(abs, J)) SSJ_H
        println("  — the :fd lane's own O(h²), not a disagreement; see this file's header.")
        println("  Diagonal of J for t = 0..min(7, T_horizon-1):")
        for i in 1:min(8, size(J, 1))
            @printf "    J[%d, %d] = %+.6f\n" (i-1) (i-1) J[i, i]
        end
        println("  First column of J (response to a date-0 r shock):")
        for i in 1:min(6, size(J, 1))
            @printf "    J[%d, 0] = %+.6f\n" (i-1) J[i, 1]
        end
        println("  — positive and slowly decaying: a higher date-0 return raises savings on")
        println("    impact, and the extra wealth is spent down over the following periods.")
    end

    # 4. The permanent-shock comparative static, the one :fd-only service
    verbose && println("\nComputing the steady-state gradient (:fd only — see the header)…")
    ∂ss = compute_steady_state_gradient(hh, env_ss; pair..., h = SS_GRAD_H)
    verbose && @printf "  ∂K_supplied/∂r = %.4f  (h = %.0e, re-solved to %.0e)\n" ∂ss[(:r, :K_supplied)] SS_GRAD_H HouseholdStages.SS_PRECONDITION_TOL

    return (; K_ss, V_ss, Λ_ss, env_ss,
             𝓔, J, J_fd, gap, ∂ss)
end

# Run when executed as a script #
#-------------------------------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running HouseholdStages sequence-space derivatives demo on the 3-stage chain…")
    @time res = ssj_demo(T_horizon = 30)
    println("\nDone.")
end
