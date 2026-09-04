#################################################################
# Becker marriage capital — finite-horizon backward + forward     #
#################################################################

# Partial equilibrium (the value of match surplus `R` is exogenous), so there is no
# market to clear. The "outer loop" is the within-match life-cycle solve itself: a
# finite-horizon backward induction over the `CapitalInvestmentStage(:match_capital) ∘
# MarkovStage(:married | match_capital)` household block, then a forward cohort
# sweep. Both loops are DRIVER logic — the duration-specific investment efficiency
# is threaded through `env` here, never inside a household stage. The inner per-age V
# backward and Λ push come from the library's stage verbs (`backward!` / `forward!`).
# See `model.jl` for the two-stage block.
#
# Backward: from a zero terminal continuation, sweep durations N_age … 1, seating
# each period's match-investment policy `m ↦ m'(m)` at its env
# `(; R, a = efficiency_at_age(t))`. The separation `MarkovStage`'s backward weights
# the married continuation by `stay(m)` (higher match capital ⇒ more future value,
# the demand-for-match-capital margin), with the separated state absorbing at value 0.
# Forward: a couple "born" married at `m0` (a point mass on the married slice) is
# pushed duration 1 … N_age. Each period the separation `MarkovStage` moves a
# `1−stay(m)` share of the married mass into the absorbing separated state, so the
# married sub-mass traces the separation curve over the relationship. Total mass on
# the `(match_capital, married)` grid is conserved (it accumulates in `separated`);
# the married marginal is the surviving couples' match-capital distribution.

include("model.jl")

using Printf

"""
Solve the Becker marriage-capital within-match life cycle by finite-horizon
backward induction + forward cohort simulation at the exogenous surplus value `R`.
Returns the duration profiles of the married rate (`married_by_age`) and mean match
capital among still-married couples (`mean_capital_by_age`), the cross-section panel
`Λ_panel` (`N_m × 2 × N_age`, one slab per duration), the per-age value arrays
`V_by_age`, and lifetime moments.
"""
function marriage_life_cycle(p = marriage_params; verbosity = 1)
    hh      = marriage_household(p)
    m_axis  = GriddedContinuous(p.m_min, p.m_max, p.N_m)
    m_grid  = collect(Float64, axisvalues(m_axis))
    cells   = cell_array(end_layout(hh))                 # (N_m, 2) cells: (:match_capital, :married)

    env_at(t) = (; R = p.R, a = efficiency_at_age(t, p))

    # Backward induction: store each duration's continuation V (V_by_age[t] is the
    # value at the START of duration t). Terminal continuation past the last is 0.
    V_by_age = Vector{Array{Float64}}(undef, p.N_age)
    V_next   = zeros(p.N_m, 2)
    for t in p.N_age:-1:1
        V_next       = copy(backward!(hh, V_next, env_at(t)))
        V_by_age[t]  = V_next
    end

    # Forward cohort sweep: a unit point mass "born" MARRIED at the grid point
    # nearest m0, pushed through each period's seated policy + separation draw.
    # Re-seating the duration-t kernel via a cheap re-backward at env_at(t)
    # guarantees the right policy.
    i0 = argmin(abs.(m_grid .- p.m0))
    Λ  = zeros(p.N_m, 2); Λ[i0, 1] = 1.0                    # newlywed: married (col 1), capital ≈ m0
    Λ_panel = zeros(p.N_m, 2, p.N_age)
    for t in 1:p.N_age
        Λ_panel[:, :, t] = Λ                                # duration-t marginal (start of duration t)
        backward!(hh, t < p.N_age ? V_by_age[t + 1] : zeros(p.N_m, 2), env_at(t))
        Λ = copy(forward!(hh, Λ))                           # invest m↦m'(m), then draw separation
    end

    married_mass        = vec(sum(Λ_panel[:, 1, :]; dims = 1))                # mass married by duration
    married_by_age      = married_mass                                       # cohort starts at mass 1
    capital_mass_married= vec(sum(m_grid .* Λ_panel[:, 1, :]; dims = 1))      # ∫ m dΛ over married
    mean_capital_by_age = capital_mass_married ./ max.(married_mass, eps())   # per-couple among married
    expected_duration   = sum(married_mass)                                  # Σ_t survival = expected relationship length

    if verbosity > 0
        total_mass = vec(sum(Λ_panel; dims = (1, 2)))
        @printf "Becker marriage-capital life cycle (N_age = %d, γ = %.2f, δ = %.2f, R = %.2f)\n" p.N_age p.γ p.δ p.R
        @printf "  total grid mass (every duration) = %.6f … %.6f (conserved)\n" minimum(total_mass) maximum(total_mass)
        @printf "  V finite everywhere              = %s\n" all(all.(isfinite, V_by_age))
        @printf "  married rate: dur 1 / mid / end  = %.4f / %.4f / %.4f\n" married_by_age[1] married_by_age[cld(p.N_age, 2)] married_by_age[end]
        @printf "  separation hazard 1−stay(m): low m=%.2f / high m=%.2f = %.4f / %.4f\n" m_grid[1] m_grid[end] (1 - retention_prob(m_grid[1], p)) (1 - retention_prob(m_grid[end], p))
        @printf "  mean match capital (married): marriage = %.3f, end = %.3f\n" mean_capital_by_age[1] mean_capital_by_age[end]
        @printf "  peak mean match capital (married)= %.3f (duration %d)\n" maximum(mean_capital_by_age) argmax(mean_capital_by_age)
        @printf "  expected relationship length     = %.3f periods\n" expected_duration
    end
    return (; married_by_age, mean_capital_by_age, married_mass, expected_duration,
              Λ_panel, V_by_age, m_grid, cells)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Becker marriage-specific capital within-match life cycle…")
    @time marriage_life_cycle()
end
