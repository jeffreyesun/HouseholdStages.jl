####################################################################
# Catherine life-cycle solve — boom vs recession comparison         #
####################################################################

# A life-cycle solve is a single backward sweep over ages (`a = N…1`) plus a
# forward cohort simulation (`a = 1…N`) — no fixed-point iteration. The
# aggregate state `z` is fixed over the cohort's life, so we run the whole
# solve twice (every age facing the boom transition, then every age facing the
# recession transition) on the SAME household object and compare. The income
# `MarkovStage`'s env-closure re-seats its transition whenever `env.z` changes,
# so the two solves differ only through the conditional variance of income.
#
# All outer-loop logic is example-side; the household block (`model.jl`) is
# `replicate_age(…)` of existing stages with no bespoke stage. The θ*(age)
# reading is identical to the CGM example (portfolio leaf, post-savings
# weighting).

include("model.jl")

using Printf

"""
Cohort-weighted mean risky share θ*(age) (cf. `cocco_gomes_maenhout`). θ*[cell]
is indexed by post-savings wealth `b'`, so each age's start-of-period
distribution is pushed forward through the leading sub-stages (shock → receipt
→ savings, `stages[1:end-1]`) onto the portfolio input before averaging.
"""
function risky_share_profile(comp, Λ_stack, N)
    profile = zeros(N)
    for a in 1:N
        leading = comp[a].buffer.stages[1:end-1]
        Λp = copy(Λ_stack[:, :, a:a])
        for s in leading
            Λp = forward!(s, Λp)
        end
        θ = HouseholdStages.policy(comp[a].buffer.stages[end])
        mass = sum(Λp)
        profile[a] = mass > 0 ? sum(θ .* Λp) / mass : NaN
    end
    return profile
end

"""
Solve the Catherine life-cycle portfolio household at a FIXED aggregate state
`z` (`:boom` / `:recession`) by backward induction + forward cohort simulation.
The income transition is re-seated to `catherine_T(z)` through the env-closure;
newborns draw from that state's stationary income distribution and enter with
the `w0_init` financial endowment. Returns the stacked `V`/`Λ`, the
cross-sectional mean wealth, the per-age mean-wealth profile, and the per-age
risky-share profile θ*(age).
"""
function catherine_solve(z::Symbol, p = catherine_params; hh = catherine_household(p), verbosity = 1)
    product = hh.buffer.stages[1]
    comp = product.buffer.components
    out_layout = product.buffer.output_layout
    nw, nε, N = layout_size(input_layout(comp[1]))[1], length(p.y_grid), p.N

    env_age(a) = (; y = age_earnings(a, p), z)

    # Backward induction — V_{N+1} = 0, no bequest #
    #---------------------------------------------#
    V_stack = zeros(nw, nε, N)
    V_next  = zeros(nw, nε, 1)
    for a in N:-1:1
        V_a = backward!(comp[a], V_next, env_age(a))
        V_stack[:, :, a] .= dropdims(V_a; dims = 3)
        V_next = copy(V_a)
    end

    # Forward cohort simulation — newborns at age 1 #
    #----------------------------------------------#
    π0 = income_stationary(z, p)
    in_cells = cell_array(input_layout(comp[1]))
    w_grid = getproperty.(in_cells[:, 1, 1], :wealth)
    i0 = argmin(abs.(w_grid .- p.w0_init))
    Λ_cohort = zeros(nw, nε, 1)
    Λ_cohort[i0, :, 1] .= π0
    Λ_stack = zeros(nw, nε, N)
    for a in 1:N
        Λ_stack[:, :, a] .= dropdims(Λ_cohort; dims = 3)
        Λ_cohort = copy(forward!(comp[a], Λ_cohort))
    end

    # Moments and age profiles #
    #--------------------------#
    m = compute_moments(hh, Λ_stack, env_age(p.peak_age))
    mean_wealth_xsec = m.mean_wealth / N
    cells = cell_array(out_layout)
    age_mean_wealth = [sum(getproperty.(cells[:, :, a], :wealth) .* Λ_stack[:, :, a]) for a in 1:N]
    θ_profile = risky_share_profile(comp, Λ_stack, N)

    if verbosity > 0
        @printf "  z = %-10s : mean wealth (x-sec) = %.4f, θ* young/peak/old = %.3f / %.3f / %.3f\n" string(z) mean_wealth_xsec θ_profile[1] maximum(θ_profile) θ_profile[N-1]
    end
    return (; z, V = V_stack, Λ = Λ_stack, mean_wealth = mean_wealth_xsec, age_mean_wealth, θ_profile)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    p  = catherine_params
    hh = catherine_household(p)
    println("Catherine (2022): countercyclical risk + life-cycle portfolio — boom vs recession")
    @printf "  σ_recession = %.2f > σ_boom = %.2f (ρ = %.2f fixed), CRRA σ = %.1f, premium = %.3f, Merton ≈ %.3f\n" p.σ_recession p.σ_boom p.ρ p.σ (sum(p.p_risky .* p.R_risky) - p.R_f) ((sum(p.p_risky .* p.R_risky) - p.R_f) / (p.σ * (sum(p.p_risky .* (p.R_risky .- p.R_f).^2) - (sum(p.p_risky .* p.R_risky) - p.R_f)^2)))
    @time begin
        boom = catherine_solve(:boom,      p; hh)
        rec  = catherine_solve(:recession, p; hh)
    end

    println("\n  age :  θ*(boom)  θ*(rec)  Δθ   | wealth(boom)  wealth(rec)")
    for a in 1:p.N
        @printf "  %3d :   %.3f     %.3f   %+.3f |   %6.3f       %6.3f\n" a boom.θ_profile[a] rec.θ_profile[a] (rec.θ_profile[a] - boom.θ_profile[a]) boom.age_mean_wealth[a] rec.age_mean_wealth[a]
    end
    @printf "\n  mean wealth : boom %.4f, recession %.4f  (Δ = %+.4f, %.1f%% more self-insurance in recession)\n" boom.mean_wealth rec.mean_wealth (rec.mean_wealth - boom.mean_wealth) (100 * (rec.mean_wealth / boom.mean_wealth - 1))
end
