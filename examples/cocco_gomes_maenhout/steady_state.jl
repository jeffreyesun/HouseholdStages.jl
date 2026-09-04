###################################################################
# CGM life-cycle solve — finite-horizon backward + forward cohort  #
###################################################################

# Identical in shape to `examples/life_cycle/steady_state.jl`: a single
# backward sweep over ages (`a = N…1`) threading the continuation value,
# then a forward cohort simulation (`a = 1…N`). Both are example outer-loop
# logic driving the ProductStage's per-age components — the household block
# itself (`model.jl`) is `replicate_age(…)` of existing stages, no bespoke
# stage.
#
# The one CGM-specific addition: each per-age component is the four-stage
# chain `… ∘ Portfolio`, so it carries TWO policy-bearing leaves (savings
# and the risky share). `policy(::ChainStage)` errors on >1 leaf, so we read
# the risky-share policy θ*(age) DIRECTLY off the portfolio leaf, the last
# stage of each age component (`comp[a].buffer.stages[end]`). The age profile
# of θ* is the headline output.

include("model.jl")

using Printf

"""
Cohort-weighted mean risky share θ*(age). The portfolio policy θ*[cell] is
indexed by POST-savings wealth `b'` (the coordinate the portfolio stage sees),
so the economically correct weight is the distribution of agents over `b'`, not
the start-of-period wealth in `Λ_stack`. For each age `a` we therefore push the
start-of-age distribution forward through the leading sub-stages of age `a`'s
component (shock → receipt → savings, i.e. `stages[1:end-1]`) to land on the
portfolio input, then average θ*[b'] against that distribution. Without this
re-timing, newborns (all mass at `b = 0`, where θ* is the indeterminate
zero-investment share) spuriously read a near-zero share. Returns the length-`N`
profile of the average equity share of invested wealth at each age.
"""
function risky_share_profile(comp, Λ_stack, N)
    profile = zeros(N)
    for a in 1:N
        leading = comp[a].buffer.stages[1:end-1]                 # shock ∘ receipt ∘ savings
        Λp = copy(Λ_stack[:, :, a:a])                            # start of age a, (nw, nε, 1)
        for s in leading
            Λp = forward!(s, Λp)                                 # → distribution at portfolio input b'
        end
        θ = HouseholdStages.policy(comp[a].buffer.stages[end])   # portfolio leaf, shape (nw, nε, 1)
        mass = sum(Λp)
        profile[a] = mass > 0 ? sum(θ .* Λp) / mass : NaN
    end
    return profile
end

"""
Solve the finite-horizon CGM life-cycle portfolio household by backward
induction + forward cohort simulation (cf. `life_cycle/steady_state.jl`).

Backward: with no bequest (`V_{N+1} = 0`), sweep `a = N…1`, feeding age-(a+1)'s
continuation value into age-a's component; this seats both the savings policy
and the risky-share policy on each age's buffer.

Forward: a unit cohort of newborns (zero financial wealth, ergodic income) is
pushed `a = 1…N`; each age-slice carries mass 1, so the cross-sectional mean
wealth is `mean_wealth / N`.

Returns the stacked `V`/`Λ`, the per-age mean-wealth profile, the per-age
cohort-weighted risky-share profile θ*(age), and the cross-sectional mean
wealth.
"""
function cgm_solve(p = cgm_params; verbosity = 1)
    hh   = cgm_household(p)
    product = hh.buffer.stages[1]               # define_moments! wraps the product in a singleton chain
    comp = product.buffer.components            # the N per-age (wealth, income, age=1) chains
    out_layout = end_layout(product)
    nw, nε, N = layout_size(start_layout(comp[1]))[1], length(p.ε_grid), p.N

    env_age(a) = (; y = age_earnings(a, p))     # portfolio returns are stage params; only y is env-borne

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
    π0 = income_stationary(p)
    in_cells = cell_array(start_layout(comp[1]))
    w_grid = getproperty.(in_cells[:, 1, 1], :wealth)
    i0 = argmin(abs.(w_grid .- p.w0_init))       # grid point nearest the newborn endowment
    Λ_cohort = zeros(nw, nε, 1)
    Λ_cohort[i0, :, 1] .= π0                      # newborns: endowment w0_init, ergodic income
    Λ_stack = zeros(nw, nε, N)
    for a in 1:N
        Λ_stack[:, :, a] .= dropdims(Λ_cohort; dims = 3)
        Λ_cohort = copy(forward!(comp[a], Λ_cohort))
    end

    # Moments and age profiles #
    #--------------------------#
    env_for_moment = env_age(p.peak_age)         # moment integrand is :wealth; env unused by it
    m = compute_moments(hh, Λ_stack, env_for_moment)
    mean_wealth_xsec = m.mean_wealth / N

    cells = cell_array(out_layout)
    age_mean_wealth = [sum(getproperty.(cells[:, :, a], :wealth) .* Λ_stack[:, :, a]) for a in 1:N]
    θ_profile = risky_share_profile(comp, Λ_stack, N)

    if verbosity > 0
        merton = (sum(p.p_risky .* p.R_risky) - p.R_f) /
                 (p.σ * (sum(p.p_risky .* (p.R_risky .- p.R_f).^2) - (sum(p.p_risky .* p.R_risky) - p.R_f)^2))
        @printf "CGM life-cycle portfolio (N = %d, σ = %.1f, β = %.3f, premium = %.3f)\n" N p.σ p.β (sum(p.p_risky .* p.R_risky) - p.R_f)
        @printf "  per-age cohort mass    : min %.6f, max %.6f (target 1.0)\n" minimum(sum(Λ_stack; dims = (1, 2))) maximum(sum(Λ_stack; dims = (1, 2)))
        @printf "  V finite               : %s\n" all(isfinite, V_stack)
        @printf "  mean wealth (x-section): %.4f\n" mean_wealth_xsec
        @printf "  Merton interior share  : %.3f (cap = %.2f)\n" merton last(p.share_bounds)
        println("  age : risky share θ*(age) | mean financial wealth")
        for a in 1:N
            bar = repeat("█", round(Int, 40 * θ_profile[a]))
            @printf "  %3d :  %.3f  %-41s %.3f\n" a θ_profile[a] bar age_mean_wealth[a]
        end
        @printf "  θ*  young (age 1) = %.3f  →  old (age %d) = %.3f   (Δ = %+.3f)\n" θ_profile[1] N θ_profile[N] (θ_profile[N] - θ_profile[1])
    end

    return (; V = V_stack, Λ = Λ_stack,
              mean_wealth = mean_wealth_xsec,
              age_mean_wealth, θ_profile)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving finite-horizon CGM life-cycle portfolio household…")
    @time cgm_solve()
end
