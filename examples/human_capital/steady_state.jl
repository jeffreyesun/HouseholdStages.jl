############################################################
# Ben-Porath life cycle — finite-horizon backward + forward #
############################################################

# Partial equilibrium (the rental rate `R` is exogenous), so there is no market
# to clear. The "outer loop" is the life-cycle solve itself: a finite-horizon
# backward induction over the single `CapitalInvestmentStage` household block, then a
# forward cohort sweep. Both loops are DRIVER logic — the age-specific ability
# profile is threaded through `env` here, never inside a household stage. The
# inner per-age V backward and Λ push come from the library's stage verbs
# (`backward!` / `forward!`). See `model.jl` for the one-stage block.
#
# Backward: from a zero terminal continuation, sweep ages N_age … 1, seating each
# age's policy `h ↦ h'(h)` at its env `(; R, a = ability_at_age(t))`.
# Forward: a cohort born at `h0` (a point mass) is pushed age 1 … N_age through
# those policies; the age-t marginal is the human-capital distribution of the
# age-t cross-section. The marginal over ALL ages is the hump-shaped life-cycle
# human-capital profile — the non-degenerate cross-section the model exists to
# produce.

include("model.jl")

using Printf

"""
Solve the Ben-Porath life cycle by finite-horizon backward induction + forward
cohort simulation at the exogenous rental rate `R`. Returns the age profile of
mean human capital (`mean_h_by_age`), the lifetime cross-section `Λ_panel`
(`N_h × N_age`, one column per age, each column a unit-mass age marginal), the
per-age value arrays `V_by_age`, and the lifetime moments.
"""
function human_capital_life_cycle(p = human_capital_params; verbosity = 1)
    hh     = human_capital_household(p)
    h_axis = GriddedContinuous(p.h_min, p.h_max, p.N_h)
    h_grid = collect(Float64, axisvalues(h_axis))

    env_at(t) = (; R = p.R, a = ability_at_age(t, p))

    # Backward induction: store each age's continuation V (V_by_age[t] is the
    # value at the START of age t). Terminal continuation past the last age is 0.
    V_by_age = Vector{Array{Float64}}(undef, p.N_age)
    V_next   = zeros(p.N_h)
    for t in p.N_age:-1:1
        V_next       = copy(backward!(hh, V_next, env_at(t)))
        V_by_age[t]  = V_next
    end

    # Forward cohort sweep: a unit point mass at the grid point nearest h0,
    # pushed through each age's seated policy. Re-seating the age-t kernel via a
    # cheap re-backward at env_at(t) guarantees the right policy for each push.
    i0      = argmin(abs.(h_grid .- p.h0))
    Λ       = zeros(p.N_h); Λ[i0] = 1.0
    Λ_panel = zeros(p.N_h, p.N_age)
    for t in 1:p.N_age
        Λ_panel[:, t] = Λ                                   # age-t marginal (start of age t)
        backward!(hh, t < p.N_age ? V_by_age[t + 1] : zeros(p.N_h), env_at(t))
        Λ = copy(forward!(hh, Λ))                           # push h ↦ h'(h) for this age
    end

    mean_h_by_age  = vec(sum(h_grid .* Λ_panel; dims = 1))
    lifetime_mean_h = sum(mean_h_by_age) / p.N_age
    earnings_by_age = p.R .* mean_h_by_age
    peak_age        = argmax(mean_h_by_age)

    if verbosity > 0
        @printf "Ben-Porath life cycle (N_age = %d, γ = %.2f, δ = %.2f, R = %.2f)\n" p.N_age p.γ p.δ p.R
        @printf "  cohort mass (every age) = %.6f … %.6f\n" minimum(sum(Λ_panel; dims = 1)) maximum(sum(Λ_panel; dims = 1))
        @printf "  V finite everywhere     = %s\n" all(all.(isfinite, V_by_age))
        @printf "  h at birth  (age 1)     = %.3f\n" mean_h_by_age[1]
        @printf "  h at peak   (age %2d)    = %.3f\n" peak_age mean_h_by_age[peak_age]
        @printf "  h at last   (age %2d)    = %.3f\n" p.N_age mean_h_by_age[end]
        @printf "  lifetime mean h         = %.3f\n" lifetime_mean_h
        @printf "  lifetime mean earnings  = %.3f\n" sum(earnings_by_age) / p.N_age
    end
    return (; mean_h_by_age, earnings_by_age, lifetime_mean_h, peak_age,
              Λ_panel, V_by_age, h_grid)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Ben-Porath human-capital life cycle…")
    @time human_capital_life_cycle()
end
