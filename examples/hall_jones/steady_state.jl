######################################################################
# Hall–Jones — finite-horizon cohort + income comparative statics     #
######################################################################

# Partial equilibrium (r, w exogenous), so no market to clear. The solve is a
# finite-horizon backward induction over the health-spending block, then a forward
# cohort sweep — the examples/health driver, here extended with the wealth/
# consumption margin and value-of-life flow. Both loops are DRIVER logic; the
# income LEVEL `env.w` is threaded through env, and the Hall–Jones headline is read
# off a comparative-statics sweep over `env.w`: richer cohorts spend a higher share
# of income on health.

include("model.jl")

using Printf

"""
Solve the Hall–Jones health-spending cohort at income level `w` by finite-horizon
backward induction + forward cohort simulation. Returns the per-age mean health
and mean wealth among the living, the per-age survival mass, life expectancy, the
reconstructed per-age medical-spending share of income, and the lifetime
health-spending share.
"""
function hall_jones_cohort(w::Real, p = hall_jones_params; verbosity = 0)
    hh    = hall_jones_household(p)
    cells = cell_array(end_layout(hh))                       # (N_w, N_h, 2, 1)
    hgrid = collect(range(p.h_min, p.h_max; length = p.N_h))
    wgrid = collect(Float64, axisvalues(GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log)))
    sz    = size(cells)

    env = (; r = p.r, w = float(w))

    # Backward induction — zero terminal continuation #
    V_by_age = Vector{Array{Float64}}(undef, p.N_age)
    V_next   = zeros(sz)
    for t in p.N_age:-1:1
        V_next      = copy(backward!(hh, V_next, env))
        V_by_age[t] = V_next
    end

    # Forward cohort sweep — newborn alive at (w0, h0) #
    iw0 = argmin(abs.(wgrid .- p.w0))
    ih0 = argmin(abs.(hgrid .- p.h0))
    Λ   = zeros(sz); Λ[iw0, ih0, 1, 1] = 1.0
    Λ_panel = zeros(sz..., p.N_age)
    for t in 1:p.N_age
        Λ_panel[:, :, :, :, t] = Λ
        backward!(hh, t < p.N_age ? V_by_age[t + 1] : zeros(sz), env)
        Λ = copy(forward!(hh, Λ))
    end

    # Per-age means among the living (the alive slice, col 1) #
    health_of = getproperty.(cells, :health)
    wealth_of = getproperty.(cells, :wealth)
    alive_mass = [sum(Λ_panel[:, :, 1, 1, t])                                   for t in 1:p.N_age]
    mean_h     = [sum(health_of[:, :, 1, 1] .* Λ_panel[:, :, 1, 1, t]) / max(alive_mass[t], eps()) for t in 1:p.N_age]
    mean_w     = [sum(wealth_of[:, :, 1, 1] .* Λ_panel[:, :, 1, 1, t]) / max(alive_mass[t], eps()) for t in 1:p.N_age]
    life_exp   = sum(alive_mass)

    # Reconstruct the medical-spending share of income from the realized health
    # path: chosen h'_t = health at the start of age t+1 (commit writes it, survival
    # leaves it). Income = labour w·y + capital income r·wealth.
    shares = Float64[]
    for t in 1:p.N_age-1
        spend  = hj_medical_cost(mean_h[t + 1], mean_h[t], p)
        income = env.w * p.y + p.r * mean_w[t]
        income > 0 && push!(shares, spend / income)
    end
    lifetime_share = isempty(shares) ? 0.0 : sum(shares) / length(shares)

    if verbosity > 0
        @printf "  w = %5.2f | life exp = %5.2f | mean h (mid) = %5.2f | health share = %.3f\n" float(w) life_exp mean_h[cld(p.N_age, 2)] lifetime_share
    end
    return (; w = float(w), alive_mass, mean_h, mean_w, life_exp,
              shares, lifetime_share, V_finite = all(all.(isfinite, V_by_age)))
end


# Driver — comparative statics over the income level #
#----------------------------------------------------#

"""
Run the Hall–Jones income comparative statics: solve the cohort at a ladder of
income levels and report the health-spending share of income at each, confirming
the share rises with income (the value-of-life-as-luxury result).
"""
function hall_jones_value_of_life(p = hall_jones_params; w_ladder = [1.0, 2.0, 4.0, 8.0], verbosity = 1)
    results = [hall_jones_cohort(w, p) for w in w_ladder]
    if verbosity > 0
        @printf "Hall–Jones value of life (N_age = %d, σ = %.1f, vsl = %.2f)\n" p.N_age p.σ p.vsl
        @printf "  V finite everywhere (all runs) = %s\n" all(r -> r.V_finite, results)
        @printf "  cohort starts at mass 1; alive-mass decays along survival curve.\n"
        @printf "  %-10s %-12s %-12s %-14s\n" "income w" "life exp" "mean h(mid)" "health share"
        for r in results
            @printf "  %-10.2f %-12.3f %-12.3f %-14.4f\n" r.w r.life_exp r.mean_h[cld(p.N_age, 2)] r.lifetime_share
        end
        shares = [r.lifetime_share for r in results]
        @printf "  health-spending share rises with income = %s (%.4f → %.4f)\n" issorted(shares) shares[1] shares[end]
    end
    return results
end


if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Hall–Jones health-spending cohort across income levels…")
    @time hall_jones_value_of_life()
end
