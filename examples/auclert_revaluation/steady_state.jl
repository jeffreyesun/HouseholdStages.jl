###################################################################
# Auclert revaluation — SS solve + one-pass revaluation demo       #
###################################################################

# Two pieces, both at fixed exogenous prices (partial equilibrium):
#
#   1. STEADY STATE. With `q = q_last` the `AssetPriceChangeStage` is inert,
#      so the model is the plain Bewley/Aiyagari fixed point. A single
#      `solve_steady_state_given_env!` confirms the block solves and the
#      revaluation stage does nothing at the SS (the consistency check the
#      channel must pass).
#
#   2. REVALUATION DEMONSTRATION (custom outer-loop snippet). The channel
#      lives OUT of steady state, at `q ≠ q_last`. We take the stationary
#      `Λ_ss` and:
#        (a) report the aggregate balance-sheet revaluation transfer
#            `∫ (q − q_last)·wealth dΛ_ss = (q − q_last)·A_mean` — the Fisher
#            redistribution to asset holders from a price move; and
#        (b) run ONE backward+forward pass of the chain at the perturbed `q`
#            (re-seating policy at the new price, pushing `Λ_ss` one period)
#            and compare next-period mean wealth against the same pass at the
#            unperturbed price, isolating the revaluation leg.
#
# Both use only the public stage verbs (`backward!`, `forward!`) and the SS
# moment — no new stage, no reaching into kernel internals.

include("model.jl")

using Printf

"Mean wealth `∫ wealth dΛ` for a distribution on the household block's end layout."
function mean_wealth(hh, Λ)
    cells = cell_array(end_layout(hh))
    return sum(getproperty.(cells, :wealth) .* Λ)
end

"""
Solve the Auclert SS (`q = q_last`, revaluation inert), then demonstrate the
revaluation channel at a perturbed price `q = q_last·(1 + dq)`.

Reports: SS aggregate wealth `A_mean`; the analytic aggregate balance-sheet
revaluation `(q − q_last)·A_mean`; and the one-period change in next-period
mean wealth induced by activating the revaluation (perturbed vs unperturbed
single pass of `Λ_ss`). The two should agree to first order — the analytic
transfer is the period-0 direct effect the within-period stage produces.
"""
function auclert_revaluation_demo(p = auclert_params; dq = 0.05, verbosity = 1)
    hh = auclert_household(p)

    # 1. Steady state — q = q_last, revaluation inert #
    #------------------------------------------------#
    env_ss = auclert_env(p)                       # q = q_last = 1.0
    res    = solve_steady_state_given_env!(hh, env_ss)
    (; V, Λ, history) = res
    A_mean = compute_moments(hh, Λ, env_ss).A_mean
    w_ss   = mean_wealth(hh, Λ)

    # 2a. Analytic aggregate revaluation transfer at q ≠ q_last #
    #----------------------------------------------------------#
    q_last = env_ss.q
    q      = q_last * (1 + dq)
    analytic_revaluation = (q - q_last) * A_mean   # = ∫ (q−q_last)·wealth dΛ_ss

    # 2b. One-pass perturbed vs unperturbed forward of Λ_ss #
    #------------------------------------------------------#
    # Unperturbed reference pass (q = q_last, revaluation off).
    backward!(hh, V, env_ss)
    Λ_next_ref = forward!(hh, copy(Λ))
    w_next_ref = mean_wealth(hh, Λ_next_ref)

    # Perturbed pass (q ≠ q_last, revaluation on) from the SAME Λ_ss and V.
    env_pert = auclert_env(p; q = q, q_last = q_last)
    backward!(hh, V, env_pert)
    Λ_next_pert = forward!(hh, copy(Λ))
    w_next_pert = mean_wealth(hh, Λ_next_pert)

    Δw_next = w_next_pert - w_next_ref

    if verbosity > 0
        @printf "Auclert revaluation channel (σ = %.1f, r = %.3f, β = %.3f)\n" p.σ p.r p.β
        @printf "  -- steady state (q = q_last, revaluation inert) --\n"
        @printf "  VFI iters = %d, Λ iters = %d\n" history.vfi_iters history.lambda_iters
        @printf "  ΣΛ                       = %.6f\n" sum(Λ)
        @printf "  V finite                 = %s\n"  all(isfinite, V)
        @printf "  aggregate wealth A_mean  = %.4f\n" A_mean
        @printf "  SS mean wealth (check)   = %.4f\n" w_ss
        @printf "  -- revaluation at q = q_last·(1+%.0f%%) = %.4f --\n" 100dq q
        @printf "  analytic transfer (q−q_last)·A_mean = %.4f\n" analytic_revaluation
        @printf "  Δ next-period mean wealth (pert−ref) = %.4f\n" Δw_next
        @printf "  ratio Δw / analytic                 = %.3f\n" Δw_next / analytic_revaluation
    end

    return (; V, Λ, A_mean, analytic_revaluation, Δw_next,
              w_next_ref, w_next_pert, history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Auclert revaluation steady state + channel demo…")
    @time res = auclert_revaluation_demo()
    println(sum(res.Λ) ≈ 1 ? "Stationary Λ sums to 1." : "WARNING: ΣΛ = $(sum(res.Λ))")
end
