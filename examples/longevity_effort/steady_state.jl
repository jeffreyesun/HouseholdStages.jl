####################################################################
# Longevity-effort life cycle — finite-horizon backward + forward   #
####################################################################

# Partial equilibrium (the return `r` and income `y` are exogenous), so there is
# no market to clear. The "outer loop" is the life-cycle solve itself: a
# finite-horizon backward induction over the `Survival ∘ Receipt ∘
# ConsumptionSavings` household block, then a forward cohort sweep. Both loops
# are DRIVER logic — the inner per-age V backward and Λ push come from the
# library's stage verbs (`backward!` / `forward!`). See `model.jl` for the block.
#
# Backward: from a zero terminal continuation, sweep ages N_age … 1. The survival
# `RetentionStage`'s backward sets `V_alive_start = c*(V_alive_after)` (death value
# = 0), so the seated survival weight `θ*(w) = clamp(κ·V_alive_after(w), 0, 1)`
# rises with the continuation value of being alive — richer / younger agents (more
# value to protect) buy more survival.
# Forward: a cohort born ALIVE at `w0` (a point mass on the alive slice) is pushed
# age 1 … N_age. Each age the survival draw moves a `1−θ(w)` share of the alive
# mass into the absorbing dead state, so the alive sub-mass traces the chosen
# survival curve. Total grid mass is conserved (it accumulates in `dead`).

include("model.jl")

using Printf

"""
Solve the longevity-effort life cycle by finite-horizon backward induction +
forward cohort simulation at the exogenous env `(; r, y)`. Returns the age
profiles of the survival rate (`survival_by_age`), mean wealth among the living
(`mean_wealth_by_age`), mean consumption among survivors (`mean_cons_by_age`),
the survival-effort policy panel `θ_panel` (`N_w × N_age`, alive slice), the
lifetime cross-section panel `Λ_panel` (`N_w × 2 × N_age`), the per-age value
arrays `V_by_age`, and life expectancy `Σ_t survival`.
"""
function longevity_life_cycle(p = longevity_params; verbosity = 1)
    hh      = longevity_household(p)
    w_axis  = GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log)
    w_grid  = collect(Float64, axisvalues(w_axis))
    env     = longevity_env(p)

    # Backward induction: V_by_age[t] is the value at the START of age t; terminal
    # continuation past the last age is 0.
    V_by_age = Vector{Array{Float64}}(undef, p.N_age)
    V_next   = zeros(p.N_w, 2)
    for t in p.N_age:-1:1
        V_next      = copy(backward!(hh, V_next, env))
        V_by_age[t] = V_next
    end

    # Forward cohort sweep: a unit point mass born ALIVE at the grid point nearest
    # w0. Re-seating the age-t policy via a cheap re-backward guarantees the right
    # survival weight θ, which we capture into θ_panel before each forward push.
    i0 = argmin(abs.(w_grid .- p.w0))
    Λ  = zeros(p.N_w, 2); Λ[i0, 1] = 1.0                       # newborn: alive (col 1), wealth ≈ w0
    Λ_panel = zeros(p.N_w, 2, p.N_age)
    θ_panel = zeros(p.N_w, p.N_age)                            # survival-effort policy, alive slice
    for t in 1:p.N_age
        Λ_panel[:, :, t] = Λ
        backward!(hh, t < p.N_age ? V_by_age[t + 1] : zeros(p.N_w, 2), env)
        θ_panel[:, t] = HouseholdStages.policy(hh.buffer.stages[1])[:, 1]   # Survival (RetentionStage) leaf
        Λ = copy(forward!(hh, Λ))                              # survive → receipt → save
    end

    alive_mass         = vec(sum(Λ_panel[:, 1, :]; dims = 1))            # mass alive at start of each age
    survival_by_age    = alive_mass                                     # cohort starts at mass 1
    wealth_mass_alive  = vec(sum(w_grid .* Λ_panel[:, 1, :]; dims = 1)) # ∫ w dΛ over the alive slice
    mean_wealth_by_age = wealth_mass_alive ./ max.(alive_mass, eps())   # per-capita among the living
    life_expectancy    = sum(alive_mass)                                # Σ_t survival = expected lifespan

    # Mean consumption among survivors via the resource identity:
    #   survivors of age t have post-receipt resources (1+r)·W_s + y·M; what they
    #   do not carry into age t+1 (= W(t+1)) is consumed. W_s, M use the seated θ.
    mean_cons_by_age = zeros(p.N_age)
    for t in 1:p.N_age
        Λa  = Λ_panel[:, 1, t]
        M   = sum(θ_panel[:, t] .* Λa)                                  # survivor mass (= alive_mass[t+1])
        W_s = sum(θ_panel[:, t] .* w_grid .* Λa)                        # survivor wealth
        W_next = t < p.N_age ? wealth_mass_alive[t + 1] : 0.0           # terminal: consume everything
        C = (1 + env.r) * W_s + env.y * M - W_next
        mean_cons_by_age[t] = M > eps() ? C / M : 0.0
    end

    if verbosity > 0
        total_mass = vec(sum(Λ_panel; dims = (1, 2)))
        @printf "Longevity-effort life cycle (N_age = %d, σ = %.1f, κ = %.3f, b̄ = %.2f, r = %.3f)\n" p.N_age p.σ p.cost_curvature p.flow_alive p.r
        @printf "  total grid mass (every age) = %.6f … %.6f (conserved, alive+dead)\n" minimum(total_mass) maximum(total_mass)
        @printf "  V finite everywhere         = %s\n" all(all.(isfinite, V_by_age))
        @printf "  survival: age 1 / mid / end = %.4f / %.4f / %.4f\n" survival_by_age[1] survival_by_age[cld(p.N_age, 2)] survival_by_age[end]
        @printf "  mean wealth (living): birth = %.3f, end = %.3f\n" mean_wealth_by_age[1] mean_wealth_by_age[end]
        @printf "  mean consumption (surv): birth = %.3f, end = %.3f\n" mean_cons_by_age[1] mean_cons_by_age[end]
        @printf "  life expectancy (Σ survival)= %.3f ages\n" life_expectancy
        # Value-of-life check: θ should rise with wealth at a fixed (young) age.
        θ1 = θ_panel[:, 1]
        @printf "  age-1 survival effort θ*(w): w_min = %.4f, w_max = %.4f (rises with wealth = %s)\n" θ1[1] θ1[end] (θ1[end] > θ1[1])
        # ... and θ should fall with age at fixed wealth (fewer years of life to protect).
        iw = argmin(abs.(w_grid .- p.w0))
        @printf "  θ* at w≈w0: age 1 = %.4f, mid = %.4f, end = %.4f (falls with age = %s)\n" θ_panel[iw, 1] θ_panel[iw, cld(p.N_age, 2)] θ_panel[iw, end] (θ_panel[iw, 1] > θ_panel[iw, end])
    end
    return (; survival_by_age, mean_wealth_by_age, mean_cons_by_age, alive_mass,
              life_expectancy, θ_panel, Λ_panel, V_by_age, w_grid)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving longevity-effort life cycle…")
    @time longevity_life_cycle()
end
