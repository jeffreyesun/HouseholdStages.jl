###############################################################
# Grossman health life cycle — finite-horizon backward + forward #
###############################################################

# Partial equilibrium (the value of health `R` is exogenous), so there is no
# market to clear. The "outer loop" is the life-cycle solve itself: a
# finite-horizon backward induction over the `CapitalInvestmentStage(:health) ∘
# MarkovStage(:alive | health)` household block, then a forward cohort sweep.
# Both loops are DRIVER logic — the age-specific health-production efficiency is
# threaded through `env` here, never inside a household stage. The inner per-age V
# backward and Λ push come from the library's stage verbs (`backward!` / `forward!`).
# See `model.jl` for the two-stage block.
#
# Backward: from a zero terminal continuation, sweep ages N_age … 1, seating each
# age's health-investment policy `h ↦ h'(h)` at its env `(; R, a = efficiency_at_age(t))`.
# The survival `MarkovStage`'s backward weights the alive continuation by
# `survival(h)` (healthier ⇒ more future value, the Grossman demand-for-health
# margin), with the dead state absorbing at value 0.
# Forward: a cohort born ALIVE at `h0` (a point mass on the alive slice) is pushed
# age 1 … N_age. Each age, the survival `MarkovStage` moves a `1−survival(h)` share
# of the alive mass into the absorbing dead state, so the alive sub-mass traces the
# survival curve over the life cycle. Total mass on the `(health, alive)` grid is
# conserved (it accumulates in `dead`); the alive marginal is the surviving
# cross-section's health distribution.

include("model.jl")

using Printf

"""
Solve the Grossman health life cycle by finite-horizon backward induction +
forward cohort simulation at the exogenous health value `R`. Returns the age
profiles of the survival rate (`survival_by_age`) and mean health among the living
(`mean_health_by_age`), the lifetime cross-section panel `Λ_panel`
(`N_h × 2 × N_age`, one slab per age), the per-age value arrays `V_by_age`, and
lifetime moments.
"""
function health_life_cycle(p = health_params; verbosity = 1)
    hh      = health_household(p)
    h_axis  = GriddedContinuous(p.h_min, p.h_max, p.N_h)
    h_grid  = collect(Float64, axisvalues(h_axis))
    cells   = cell_array(end_layout(hh))                 # (N_h, 2) cells: (:health, :alive)

    env_at(t) = (; R = p.R, a = efficiency_at_age(t, p))

    # Backward induction: store each age's continuation V (V_by_age[t] is the value
    # at the START of age t). Terminal continuation past the last age is 0.
    V_by_age = Vector{Array{Float64}}(undef, p.N_age)
    V_next   = zeros(p.N_h, 2)
    for t in p.N_age:-1:1
        V_next       = copy(backward!(hh, V_next, env_at(t)))
        V_by_age[t]  = V_next
    end

    # Forward cohort sweep: a unit point mass born ALIVE at the grid point nearest
    # h0, pushed through each age's seated policy + survival draw. Re-seating the
    # age-t kernel via a cheap re-backward at env_at(t) guarantees the right policy.
    i0 = argmin(abs.(h_grid .- p.h0))
    Λ  = zeros(p.N_h, 2); Λ[i0, 1] = 1.0                    # newborn: alive (col 1), health ≈ h0
    Λ_panel = zeros(p.N_h, 2, p.N_age)
    for t in 1:p.N_age
        Λ_panel[:, :, t] = Λ                                # age-t marginal (start of age t)
        backward!(hh, t < p.N_age ? V_by_age[t + 1] : zeros(p.N_h, 2), env_at(t))
        Λ = copy(forward!(hh, Λ))                           # invest h↦h'(h), then survive
    end

    alive_mass         = vec(sum(Λ_panel[:, 1, :]; dims = 1))                # mass alive by age
    survival_by_age    = alive_mass                                          # cohort starts at mass 1
    health_mass_alive  = vec(sum(h_grid .* Λ_panel[:, 1, :]; dims = 1))      # ∫ h dΛ over alive
    mean_health_by_age = health_mass_alive ./ max.(alive_mass, eps())       # per-capita among living
    life_expectancy    = sum(alive_mass)                                     # Σ_t survival = expected lifespan

    if verbosity > 0
        total_mass = vec(sum(Λ_panel; dims = (1, 2)))
        @printf "Grossman health life cycle (N_age = %d, γ = %.2f, δ = %.2f, R = %.2f)\n" p.N_age p.γ p.δ p.R
        @printf "  total grid mass (every age) = %.6f … %.6f (conserved)\n" minimum(total_mass) maximum(total_mass)
        @printf "  V finite everywhere         = %s\n" all(all.(isfinite, V_by_age))
        @printf "  survival: age 1 / mid / end = %.4f / %.4f / %.4f\n" survival_by_age[1] survival_by_age[cld(p.N_age, 2)] survival_by_age[end]
        @printf "  mean health (living): birth = %.3f, end = %.3f\n" mean_health_by_age[1] mean_health_by_age[end]
        @printf "  peak mean health (living)   = %.3f (age %d)\n" maximum(mean_health_by_age) argmax(mean_health_by_age)
        @printf "  life expectancy (Σ survival)= %.3f ages\n" life_expectancy
    end
    return (; survival_by_age, mean_health_by_age, alive_mass, life_expectancy,
              Λ_panel, V_by_age, h_grid, cells)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Grossman health-capital life cycle…")
    @time health_life_cycle()
end
