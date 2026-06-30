###########################################################################
# Guvenen–Kuruşçu — per-type Ben-Porath life cycle + cross-type aggregation #
###########################################################################

# Partial equilibrium (the skill price `R` is exogenous), so there is no market to
# clear. The "outer loop" is the per-type life-cycle solve plus the cross-type
# aggregation: for each permanent ability type we run a finite-horizon backward
# induction over the single `CapitalInvestmentStage` household block, then a forward cohort
# sweep — the IDENTICAL pattern to `examples/human_capital`, threaded with that
# type's ability profile through `env`. Then we STACK the per-type cross-sections,
# weighted by population shares — the `⊕`-over-types as driver-level aggregation. The
# per-type env threading is the point: each type needs its own `env.a`, so the type
# heterogeneity is realized in the driver, not inside a household stage.
#
# Headline result (the paper's): higher-ability types face a lower effort cost of
# producing human capital, invest more, and reach higher human capital → a wider
# lifetime-earnings distribution across the population than any single type displays.

include("model.jl")

using Printf

"""
Solve ONE ability type's Ben-Porath life cycle by finite-horizon backward induction
+ forward cohort simulation at skill price `R` and the type-`k` ability profile.
Returns the age profile of mean human capital (`mean_h_by_age`), the per-age cohort
panel `Λ_panel` (`N_h × N_age`, unit-mass columns), and lifetime moments. This is
`examples/human_capital`'s solver, indexed by ability type `k`.
"""
function gk_type_life_cycle(k::Integer, p = gk_params)
    hh     = gk_household(p)
    h_axis = GriddedContinuous(p.h_min, p.h_max, p.N_h)
    h_grid = collect(Float64, axisvalues(h_axis))

    env_at(t) = (; R = p.R, a = ability_at_age(t, k, p))

    # Backward induction: V_by_age[t] is the value at the START of age t.
    V_by_age = Vector{Array{Float64}}(undef, p.N_age)
    V_next   = zeros(p.N_h)
    for t in p.N_age:-1:1
        V_next      = copy(backward!(hh, V_next, env_at(t)))
        V_by_age[t] = V_next
    end

    # Forward cohort sweep: a unit point mass at the grid point nearest h0, pushed
    # through each age's seated policy (re-seated via a cheap re-backward at env_at(t)).
    i0      = argmin(abs.(h_grid .- p.h0))
    Λ       = zeros(p.N_h); Λ[i0] = 1.0
    Λ_panel = zeros(p.N_h, p.N_age)
    for t in 1:p.N_age
        Λ_panel[:, t] = Λ
        backward!(hh, t < p.N_age ? V_by_age[t + 1] : zeros(p.N_h), env_at(t))
        Λ = copy(forward!(hh, Λ))
    end

    mean_h_by_age   = vec(sum(h_grid .* Λ_panel; dims = 1))
    earnings_by_age = p.R .* mean_h_by_age
    lifetime_mean_h = sum(mean_h_by_age) / p.N_age
    lifetime_earn   = sum(earnings_by_age)                  # PV-free lifetime earnings (sum over ages)
    return (; mean_h_by_age, earnings_by_age, lifetime_mean_h, lifetime_earn,
              Λ_panel, V_by_age, h_grid)
end

"""
Solve the Guvenen–Kuruşçu economy: run the per-type Ben-Porath life cycle for every
permanent ability type, then aggregate the per-type cross-sections weighted by
population shares (the `⊕`-over-types). Reports per-type mean human capital and
lifetime earnings, the population-weighted aggregate, and the lifetime-earnings
inequality across types (the paper's headline). Returns per-type results, the
weighted age profiles, and a between-type lifetime-earnings spread.
"""
function gk_economy(p = gk_params; verbosity = 1)
    n_types = length(p.type_names)
    per_type = [gk_type_life_cycle(k, p) for k in 1:n_types]

    # Population-weighted aggregate age profiles (the ⊕-over-types cross-section).
    agg_mean_h   = sum(p.type_share[k] .* per_type[k].mean_h_by_age   for k in 1:n_types)
    agg_earnings = sum(p.type_share[k] .* per_type[k].earnings_by_age for k in 1:n_types)

    type_lifetime_earn = [per_type[k].lifetime_earn for k in 1:n_types]
    type_lifetime_h    = [per_type[k].lifetime_mean_h for k in 1:n_types]
    # Headline inequality measure: ratio of top-type to bottom-type lifetime earnings.
    earn_ratio_hi_lo = type_lifetime_earn[end] / type_lifetime_earn[1]

    if verbosity > 0
        all_V_finite = all(all(all.(isfinite, r.V_by_age)) for r in per_type)
        all_mass_ok  = all(all(isapprox.(sum(r.Λ_panel; dims = 1), 1.0; atol = 1e-8)) for r in per_type)
        @printf "Guvenen–Kuruşçu human capital with permanent ability types\n"
        @printf "  (N_age = %d, γ = %.2f, δ = %.2f, R = %.2f, n_types = %d)\n" p.N_age p.γ p.δ p.R n_types
        @printf "  V finite (all types)        = %s\n" all_V_finite
        @printf "  cohort mass conserved (all) = %s\n" all_mass_ok
        println("  ── per ability type ──────────────────────────────────────────")
        @printf "  %-6s %7s %7s %9s %10s %12s\n" "type" "share" "a_y→a_o" "peak h" "life h̄" "life earn"
        for k in 1:n_types
            r = per_type[k]
            @printf "  %-6s %7.2f  %.2f→%.2f %9.3f %10.3f %12.3f\n" String(p.type_names[k]) p.type_share[k] p.a_young[k] p.a_old[k] maximum(r.mean_h_by_age) r.lifetime_mean_h r.lifetime_earn
        end
        println("  ──────────────────────────────────────────────────────────────")
        @printf "  population-weighted lifetime h̄  = %.3f\n" sum(agg_mean_h) / p.N_age
        @printf "  population-weighted lifetime earn = %.3f\n" sum(agg_earnings)
        @printf "  lifetime-earnings ratio high/low  = %.3f  (headline inequality)\n" earn_ratio_hi_lo
    end
    return (; per_type, agg_mean_h, agg_earnings, type_lifetime_earn, type_lifetime_h,
              earn_ratio_hi_lo)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Guvenen–Kuruşçu human capital with permanent ability types…")
    @time gk_economy()
end
