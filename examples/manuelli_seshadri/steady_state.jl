######################################################################
# Manuelli–Seshadri — finite-horizon backward + forward cohort        #
######################################################################

# Partial equilibrium (the wage is exogenous), so there is no market to clear. The
# solve is the life-cycle solve itself: a finite-horizon backward induction over the
# single `CapitalInvestmentStage` block, then a forward cohort sweep. Both are DRIVER logic
# — the schooling→work phase switch is threaded through `env` here via `env_at_age`,
# never inside a household stage. This is the `examples/human_capital` driver with a
# phase-dependent env; the schooling corner is env, not a new stage.

include("model.jl")

using Printf

"""
Solve the Manuelli–Seshadri multi-stage human-capital life cycle by finite-horizon
backward induction + forward cohort simulation. Returns the age profile of mean
human capital (`mean_h_by_age`), the gross-investment and earnings profiles, the
phase boundary, the lifetime cross-section `Λ_panel` (`N_h × N_age`), the per-age
value arrays `V_by_age`, and the human-capital grid.
"""
function manuelli_seshadri_life_cycle(p = manuelli_seshadri_params; verbosity = 1)
    hh     = manuelli_seshadri_household(p)
    h_axis = GriddedContinuous(p.h_min, p.h_max, p.N_h)
    h_grid = collect(Float64, axisvalues(h_axis))

    env_at(t) = env_at_age(t, p)

    # Backward induction: V_by_age[t] is the value at the START of age t; terminal
    # continuation past the last age is zero.
    V_by_age = Vector{Array{Float64}}(undef, p.N_age)
    V_next   = zeros(p.N_h)
    for t in p.N_age:-1:1
        V_next      = copy(backward!(hh, V_next, env_at(t)))
        V_by_age[t] = V_next
    end

    # Forward cohort: a unit point mass born at h0, pushed age 1 … N_age. Re-seating
    # the age-t policy with a cheap re-backward guarantees the right push each age.
    i0      = argmin(abs.(h_grid .- p.h0))
    Λ       = zeros(p.N_h); Λ[i0] = 1.0
    Λ_panel = zeros(p.N_h, p.N_age)
    for t in 1:p.N_age
        Λ_panel[:, t] = Λ
        backward!(hh, t < p.N_age ? V_by_age[t + 1] : zeros(p.N_h), env_at(t))
        Λ = copy(forward!(hh, Λ))
    end

    mean_h_by_age   = vec(sum(h_grid .* Λ_panel; dims = 1))
    earn_by_age     = [env_at(t).earn * mean_h_by_age[t] for t in 1:p.N_age]
    # Gross investment implied by the realized h-path: i_t = h_{t+1} − (1−δ)h_t.
    invest_by_age   = [t < p.N_age ? max(mean_h_by_age[t + 1] - (1 - p.δ) * mean_h_by_age[t], 0.0) : 0.0
                       for t in 1:p.N_age]
    peak_age        = argmax(mean_h_by_age)

    if verbosity > 0
        @printf "Manuelli–Seshadri multi-stage HC (N_age = %d, T_school = %d, γ = %.2f, δ = %.2f)\n" p.N_age p.T_school p.γ p.δ
        @printf "  cohort mass (every age)      = %.6f … %.6f\n" minimum(sum(Λ_panel; dims = 1)) maximum(sum(Λ_panel; dims = 1))
        @printf "  V finite everywhere          = %s\n" all(all.(isfinite, V_by_age))
        @printf "  h at birth   (age  1)        = %.3f\n" mean_h_by_age[1]
        @printf "  h at end of school (age %2d)  = %.3f\n" p.T_school mean_h_by_age[p.T_school]
        @printf "  h at peak    (age %2d)        = %.3f\n" peak_age mean_h_by_age[peak_age]
        @printf "  h at last    (age %2d)        = %.3f\n" p.N_age mean_h_by_age[end]
        @printf "  schooling-phase HC growth    = %.1f%% per age\n" 100 * ((mean_h_by_age[p.T_school] / mean_h_by_age[1])^(1 / max(p.T_school - 1, 1)) - 1)
        @printf "  mean investment: school/work = %.3f / %.3f\n" (sum(invest_by_age[1:p.T_school]) / p.T_school) (sum(invest_by_age[p.T_school+1:end]) / (p.N_age - p.T_school))
    end
    return (; mean_h_by_age, earn_by_age, invest_by_age, peak_age,
              T_school = p.T_school, Λ_panel, V_by_age, h_grid)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Manuelli–Seshadri multi-stage human-capital life cycle…")
    @time manuelli_seshadri_life_cycle()
end
