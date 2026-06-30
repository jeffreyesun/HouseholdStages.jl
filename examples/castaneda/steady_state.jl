###############################################################
# Castañeda steady state — backward VFI + demographics forward #
###############################################################
#
# Two pieces:
#
#   (1) Backward VFI on the BLOCK (`model.jl`). The block's `MarkovStage(:age)`
#       carries a sub-stochastic matrix whose oldest row is zero — a built-in
#       terminal condition (no continuation after certain death) — so the
#       Bellman backward iteration is a contraction and `solve_vfi_…!` runs
#       to a fixed point unchanged.
#
#   (2) A CUSTOM DEMOGRAPHICS FORWARD LOOP (this file). One pass of the block's
#       `forward!` ages every household, applies the survival hazard (deceased
#       mass bleeds off the age axis), and resolves earnings/receipt/savings.
#       The driver then closes the demographics the block deliberately does
#       NOT: it computes the deceased mass and their wealth marginal, and
#       re-injects an equal NEWBORN mass at age 1 carrying that wealth as
#       ACCIDENTAL BEQUESTS, with earnings drawn from the ergodic earnings
#       distribution. Mass injected ≡ mass deceased, so `ΣΛ = 1` is preserved
#       exactly. This is rolled by hand, mirroring the life_cycle finite-horizon
#       sweep — no bespoke stage.

include("model.jl")

using Printf

"""
Solve the Castañeda steady state: backward VFI on the block, then iterate the
custom demographics forward map to its stationary distribution.

The demographics map, per pass, is

    Λ⁺ = block.forward(Λ) ;   Λ⁺[:, :, 1] += beq(Λ) ⊗ π_earn

where `beq(Λ)[w]` is the wealth marginal of the households who die this pass —
retirees who fail the survival draw (`(1 − surv_ret)` of each retired age) plus
the entire maximum-age cohort. Their mass is injected as newborns at age 1, so
`Σ beq = ` deceased mass `= ` mass the age Markov bled off, and `ΣΛ` is fixed.
"""
function castaneda_steady_state(p = castaneda_params;
                                tol = 1e-10, max_iter = 50_000, verbosity = 1)
    hh  = castaneda_household(p)
    env = castaneda_env(p)

    # (1) Backward VFI — terminal condition baked into the age matrix's zero row.
    vfi = solve_vfi_steady_state_given_env!(hh, env)

    # (2) Custom demographics forward loop.
    nw, ne, na = p.N_w, length(p.e_grid), p.N_age
    π0 = earnings_stationary(p)
    death_haz = 1 - p.surv_ret
    retired   = p.retire_age:(na - 1)        # ages that face the survival hazard

    # Newborns start at age 1, zero wealth, ergodic earnings; iterate to stationarity.
    Λ = zeros(nw, ne, na)
    Λ[1, :, 1] .= π0

    diff, iters = Inf, 0
    while diff > tol
        # Accidental-bequest wealth marginal of this pass's deceased.
        beq = zeros(nw)
        for a in retired
            @views beq .+= death_haz .* vec(sum(Λ[:, :, a]; dims = 2))
        end
        @views beq .+= vec(sum(Λ[:, :, na]; dims = 2))   # oldest cohort: certain death

        Λ_new = forward!(hh.buffer, hh.spec, Λ)          # block forward: age, kill, earn, save
        @views Λ_new[:, :, 1] .+= beq * π0'              # re-inject newborns with bequests

        diff = maximum(abs, Λ_new .- Λ)
        Λ .= Λ_new
        iters += 1
        iters == max_iter && error("castaneda demographics loop failed to converge (diff = $diff)")
    end

    K = compute_moments(hh, Λ, env).K_supplied

    # Diagnostics: population age distribution and mean wealth by age.
    age_mass  = [sum(@view Λ[:, :, a]) for a in 1:na]
    w_grid    = axis_grid(hh.buffer.input_layout, :wealth)
    age_wealth = [age_mass[a] > 0 ? sum(w_grid .* vec(sum(@view Λ[:, :, a]; dims = 2))) / age_mass[a] : 0.0
                  for a in 1:na]

    if verbosity > 0
        @printf "Castañeda steady state (β=%.2f σ=%.1f r=%.3f, N_age=%d, retire=%d, surv_ret=%.2f)\n" p.β p.σ p.r p.N_age p.retire_age p.surv_ret
        @printf "  ΣΛ (total mass)        : %.8f\n" sum(Λ)
        @printf "  V finite               : %s  (VFI iters %d)\n" all(isfinite, vfi.V) vfi.iters
        @printf "  demographics iters     : %d  (final diff %.2e)\n" iters diff
        @printf "  aggregate wealth K     : %.4f\n" K
        @printf "  retired population share: %.4f\n" sum(age_mass[p.retire_age:end])
        @printf "  age population shares  : %s\n" join((@sprintf("%.3f", m) for m in age_mass), " ")
        @printf "  mean wealth by age     : %s\n" join((@sprintf("%.2f", x) for x in age_wealth), " ")
    end

    return (; V = vfi.V, Λ, K, age_mass, age_wealth, demo_iters = iters)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Castañeda OLG steady state…")
    @time castaneda_steady_state()
end
