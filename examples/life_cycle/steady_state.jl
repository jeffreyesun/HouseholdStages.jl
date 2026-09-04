###############################################################
# Life-cycle solve — finite-horizon backward + forward cohort #
###############################################################

# A life-cycle model is NOT a stationary steady state: there is no
# VFI-to-fixed-point. The solve is a single backward sweep over ages
# (`a = N…1`) followed by a forward cohort simulation (`a = 1…N`). Both
# are rolled here as example outer-loop logic — the household block itself
# (`model.jl`) is `replicate_age(…)` of existing stages with no bespoke
# stage. The driver reaches into the ProductStage's per-age components
# (`hh.buffer.components[a]`), feeding each age its own env (the
# age-earnings `y(age)` and the rate `r`) and threading the continuation
# value across ages — the wiring a finite-horizon problem needs and the
# block-diagonal ProductStage deliberately does not supply.

include("model.jl")

using Printf

"""
Solve the finite-horizon life-cycle household by backward induction +
forward cohort simulation.

Backward: with no bequest (`V_{N+1} = 0`), sweep `a = N…1`, feeding age-(a+1)'s
continuation value into age-a's `replicate_age` component. This seats each
age's savings policy on its own buffer.

Forward: a unit cohort of newborns (zero wealth, ergodic income) is pushed
`a = 1…N` through the per-age components; the age-a distribution is stored in
slice `a` of a stacked `(N_w, n_ε, N)` Λ. Each age-slice carries mass 1, so the
cross-sectional average over the `N` living ages is `mean_wealth / N`.

Returns the stacked `V`/`Λ` tensors, the per-age mean-wealth profile, the
cross-sectional `mean_wealth`, and the seated per-age savings policies.
"""
function life_cycle_solve(p = life_cycle_params; verbosity = 1)
    hh   = life_cycle_household(p)
    # `define_moments!` wraps the ProductStage in a singleton ChainStage, so the
    # product (and its per-age components) sits one level down.
    product = hh.buffer.stages[1]
    comp = product.buffer.components            # the N per-age (wealth,income,age=1) chains
    out_layout = end_layout(product)      # stacked (wealth, income, age=N)
    nw, nε, N = layout_size(start_layout(comp[1]))[1], length(p.ε_grid), p.N

    env_age(a) = (; r = p.r, y = age_earnings(a, p))

    # Backward induction — V_{N+1} = 0, no bequest #
    #---------------------------------------------#
    V_stack = zeros(nw, nε, N)
    V_next  = zeros(nw, nε, 1)                   # continuation value beyond the last age
    for a in N:-1:1
        V_a = backward!(comp[a], V_next, env_age(a))
        V_stack[:, :, a] .= dropdims(V_a; dims = 3)
        V_next = copy(V_a)
    end

    # Forward cohort simulation — newborns at age 1 #
    #----------------------------------------------#
    π0 = income_stationary(p)
    Λ_cohort = zeros(nw, nε, 1)
    Λ_cohort[1, :, 1] .= π0                      # newborns: zero wealth, ergodic income
    Λ_stack = zeros(nw, nε, N)
    for a in 1:N
        Λ_stack[:, :, a] .= dropdims(Λ_cohort; dims = 3)
        Λ_cohort = copy(forward!(comp[a], Λ_cohort))   # → start-of-age-(a+1) distribution
    end

    # Moments — the stacked Λ has the :age axis; the attached integrand reads it #
    #---------------------------------------------------------------------------#
    env_for_moment = env_age(p.peak_age)         # moment integrand is :wealth; env unused by it
    m = compute_moments(hh, Λ_stack, env_for_moment)
    mean_wealth_xsec = m.mean_wealth / N         # each age-slice has unit mass

    cells = cell_array(out_layout)
    age_mean_wealth = [sum(getproperty.(cells[:, :, a], :wealth) .* Λ_stack[:, :, a]) for a in 1:N]

    if verbosity > 0
        @printf "Life-cycle solve (N = %d, σ = %.1f, r = %.3f, β = %.3f)\n" N p.σ p.r p.β
        @printf "  per-age cohort mass     : min %.6f, max %.6f (target 1.0)\n" minimum(sum(Λ_stack; dims = (1, 2))) maximum(sum(Λ_stack; dims = (1, 2)))
        @printf "  V finite                : %s\n" all(isfinite, V_stack)
        @printf "  mean wealth (x-section) : %.4f\n" mean_wealth_xsec
        @printf "  wealth at age 1 / peak  : %.4f / %.4f (peak at age %d)\n" age_mean_wealth[1] maximum(age_mean_wealth) argmax(age_mean_wealth)
        @printf "  wealth at final age %d   : %.4f\n" N age_mean_wealth[N]
    end

    return (; V = V_stack, Λ = Λ_stack,
              mean_wealth = mean_wealth_xsec,
              age_mean_wealth,
              policies = [HouseholdStages.policy(comp[a]) for a in 1:N])
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving finite-horizon life-cycle household…")
    @time life_cycle_solve()
end
