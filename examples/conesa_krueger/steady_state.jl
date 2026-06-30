###############################################################
# Conesa–Krueger solve — finite-horizon OLG + PAYG balance     #
###############################################################

# An OLG model is NOT a stationary steady state: there is no
# VFI-to-fixed-point. The within-cohort solve is a single backward sweep
# over ages (`a = N…1`) followed by a forward cohort simulation
# (`a = 1…N`) — the life_cycle driver verbatim in structure. ON TOP of
# that, the unfunded PAYG social-security system must balance: the flat
# benefit `b` is pinned by `b · (#retired cohorts) = τ · (aggregate labor
# earnings)`. Because `b` enters retirees' receipt (hence the backward
# solve), this is a fixed point in `b` — a plain outer loop (the same
# status as a tatonnement on K̄), rolled here as example code. The
# household block itself (`model.jl`) is `replicate_age(…)` of existing
# stages with no bespoke stage.

include("model.jl")

using Printf

"""
Per-age `env`. The driver threads age-dependence here (the `:age` axis is a
size-1 singleton inside each `replicate_age` component). A worker
(`age ≤ retire_age`) faces net wage `(1−τ)·y(age)` and no benefit; a retiree
faces zero labor earnings and the flat SS benefit `b`.
"""
function ck_env_age(a::Integer, b::Float64, p = conesa_krueger_params)
    working = a <= p.retire_age
    netwage = working ? (1 - p.τ) * age_earnings(a, p) : 0.0
    benefit = working ? 0.0 : b
    return (; p.r, netwage, benefit)
end

"""
One finite-horizon OLG pass at a GIVEN benefit `b`: backward induction
(`V_{N+1} = 0`, no bequest) seating each age's savings policy, then a forward
cohort simulation from newborns (zero wealth, ergodic income). Each age-slice
carries unit mass. Returns the stacked `V`/`Λ`, the per-age mean-wealth profile,
the cross-sectional mean wealth, and the aggregate labor earnings `E` (used by
the PAYG balance loop to update `b`).
"""
function ck_one_pass(hh, comp, out_layout, nw, nε, N, b, p)
    # Backward induction — V_{N+1} = 0, no bequest #
    V_stack = zeros(nw, nε, N)
    V_next  = zeros(nw, nε, 1)
    for a in N:-1:1
        V_a = backward!(comp[a], V_next, ck_env_age(a, b, p))
        V_stack[:, :, a] .= dropdims(V_a; dims = 3)
        V_next = copy(V_a)
    end

    # Forward cohort simulation — newborns at age 1 #
    π0 = income_stationary(p)
    Λ_cohort = zeros(nw, nε, 1)
    Λ_cohort[1, :, 1] .= π0                      # newborns: zero wealth, ergodic income
    Λ_stack = zeros(nw, nε, N)
    for a in 1:N
        Λ_stack[:, :, a] .= dropdims(Λ_cohort; dims = 3)
        Λ_cohort = copy(forward!(comp[a], Λ_cohort))
    end

    # Aggregate labor earnings E = Σ_{working ages} Σ_cells y(a)·ε · Λ. With
    # each cohort at unit mass, E feeds the PAYG balance b = τ·E / #retirees.
    cells = cell_array(out_layout)
    E = 0.0
    for a in 1:N
        a <= p.retire_age || continue
        ya = age_earnings(a, p)
        E += ya * sum(getproperty.(cells[:, :, a], :income) .* Λ_stack[:, :, a])
    end

    env_moment = ck_env_age(p.peak_age, b, p)    # moment integrand is :wealth; env unused by it
    m = compute_moments(hh, Λ_stack, env_moment)
    mean_wealth_xsec = m.mean_wealth / N
    age_mean_wealth = [sum(getproperty.(cells[:, :, a], :wealth) .* Λ_stack[:, :, a]) for a in 1:N]

    return (; V = V_stack, Λ = Λ_stack, E,
              mean_wealth = mean_wealth_xsec, age_mean_wealth)
end

"""
Solve the Conesa–Krueger OLG steady state. With `balance = true`, iterate the
unfunded PAYG benefit `b` to budget balance — `b = τ·E / R` where `E` is
aggregate labor earnings and `R = N − retire_age` is the number of retired
cohorts (each of unit mass) — re-running the finite-horizon OLG pass each
iteration until `b` converges. With `balance = false`, the benefit is fixed at
`b0` (pure partial equilibrium). Reports the wealth profile, cross-sectional
mean wealth, and (when balancing) the equilibrium benefit and replacement rate.
"""
function conesa_krueger_solve(p = conesa_krueger_params; balance = true, b0 = 0.3,
                              tol = 1e-6, maxit = 100, verbosity = 1)
    hh      = conesa_krueger_household(p)
    product = hh.buffer.stages[1]
    comp    = product.buffer.components
    out_layout = product.buffer.output_layout
    nw, nε, N  = layout_size(input_layout(comp[1]))[1], length(p.ε_grid), p.N
    R = N - p.retire_age                          # number of retired cohorts (unit mass each)

    b   = b0
    res = ck_one_pass(hh, comp, out_layout, nw, nε, N, b, p)
    iters = 0
    if balance
        for it in 1:maxit
            iters = it
            res   = ck_one_pass(hh, comp, out_layout, nw, nε, N, b, p)
            b_new = p.τ * res.E / R
            Δ     = abs(b_new - b)
            b     = b_new
            Δ < tol && break
        end
    end

    # Average lifetime peak earnings, for the replacement rate b / (peak wage).
    peak_wage = age_earnings(p.peak_age, p) * sum(income_stationary(p) .* p.ε_grid)

    if verbosity > 0
        @printf "Conesa–Krueger OLG (N = %d, retire @ %d, σ = %.1f, r = %.3f, τ = %.2f)\n" p.N p.retire_age p.σ p.r p.τ
        @printf "  PAYG balance            : %s (%d iters)\n" (balance ? "on" : "off (fixed b)") iters
        @printf "  benefit b               : %.4f   (replacement rate b/peak-wage = %.3f)\n" b (b / peak_wage)
        @printf "  per-age cohort mass     : min %.6f, max %.6f (target 1.0)\n" minimum(sum(res.Λ; dims = (1, 2))) maximum(sum(res.Λ; dims = (1, 2)))
        @printf "  V finite                : %s\n" all(isfinite, res.V)
        @printf "  mean wealth (x-section) : %.4f\n" res.mean_wealth
        @printf "  wealth at age 1 / peak  : %.4f / %.4f (peak at age %d)\n" res.age_mean_wealth[1] maximum(res.age_mean_wealth) argmax(res.age_mean_wealth)
        @printf "  wealth at final age %d   : %.4f\n" N res.age_mean_wealth[N]
        @printf "  aggregate labor earnings: %.4f   (total benefits paid = %.4f)\n" res.E (b * R)
    end

    return (; b, V = res.V, Λ = res.Λ, mean_wealth = res.mean_wealth,
              age_mean_wealth = res.age_mean_wealth, E = res.E, balance_iters = iters,
              policies = [HouseholdStages.policy(comp[a]) for a in 1:N])
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Conesa–Krueger OLG with PAYG social security…")
    @time conesa_krueger_solve()
end
