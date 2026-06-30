###############################################################
# Livshits–MacGee–Tertilt solve — finite-horizon bankruptcy    #
###############################################################

# A life-cycle bankruptcy model is NOT a stationary steady state: the solve
# is a single backward sweep over ages (`a = N…1`) threading the
# continuation value, then a forward cohort simulation (`a = 1…N`) — the
# life_cycle driver verbatim in structure, here over the FULL six-stage
# `examples/default` chain replicated per age. Two finite-horizon points:
#
#   (i)  the per-age `env` carries the age-earnings hump `y(age)` (the
#        `:age` axis is a size-1 singleton inside each component, so
#        age-dependence is threaded by the driver, exactly as life_cycle
#        does);
#   (ii) the terminal continuation forbids DYING IN DEBT: with no bequest,
#        the period-(N+1) value is `0` for non-negative estates and `−∞`
#        for negative assets, so the last-period household either repays to
#        a non-negative position or files. This is a no-negative-estate
#        terminal CONDITION (an array the driver supplies), not a stage.
#
# The household block itself (`model.jl`) is `replicate_age(…)` of existing
# stages with no bespoke stage. Prices are exogenous (risk-free unit bond,
# partial equilibrium).

include("model.jl")

using Printf

"""
Per-age `env` for the LMT household. Threads the deterministic age-earnings
hump `y(age)` (the `:age` axis is a size-1 singleton inside each
`replicate_age` component) alongside the exogenous rate `r` and garnishment
`λ`, which the receipt closure reads.
"""
lmt_env_age(a::Integer, p = lmt_params) = (; p.r, p.λ, y = age_earnings(a, p))

"""
Solve the Livshits–MacGee–Tertilt life-cycle bankruptcy household by backward
induction + forward cohort simulation.

Backward: with a no-negative-estate terminal condition (`V_{N+1} = 0` for
`a ≥ 0`, `−∞` for `a < 0`), sweep `a = N…1`, feeding age-(a+1)'s continuation
into age-a's component; this seats each age's file/repay and savings policies.

Forward: a unit cohort of newborns (zero wealth, ergodic income, good standing)
is pushed `a = 1…N`; the age-a distribution is stored in slice `a` of a stacked
`(N_a, n_ε, 2, N)` Λ. Each age-slice carries unit mass, so cross-sectional means
are the stacked moment divided by `N`.

Reports cross-sectional mean assets, the excluded share, the by-age excluded
stock share, and the by-age filing hazard (conditional on good standing).
"""
function lmt_solve(p = lmt_params; verbosity = 1)
    hh      = lmt_household(p)
    product = hh.buffer.stages[1]
    comp    = product.buffer.components
    out_layout = product.buffer.output_layout
    in_layout  = input_layout(comp[1])
    na, nε, ns = layout_size(in_layout)[1], length(p.ε_grid), 2
    N = p.N

    # Wealth grid and the newborn (zero-wealth) index.
    cells_in = cell_array(in_layout)
    wgrid = getproperty.(cells_in[:, 1, 1, 1], :wealth)
    born_idx = argmin(abs.(wgrid))

    # Backward induction — no-negative-estate terminal condition #
    #-----------------------------------------------------------#
    V_stack = zeros(na, nε, ns, N)
    V_next  = zeros(na, nε, ns, 1)
    V_next[wgrid .< -1e-9, :, :, :] .= -1e8     # no bequest, and no dying in debt
    for a in N:-1:1
        V_a = backward!(comp[a], V_next, lmt_env_age(a, p))
        V_stack[:, :, :, a] .= dropdims(V_a; dims = 4)
        V_next = copy(V_a)
    end

    # Forward cohort simulation — newborns at age 1 #
    #----------------------------------------------#
    π0 = income_stationary(p)
    Λ_cohort = zeros(na, nε, ns, 1)
    Λ_cohort[born_idx, :, 1, 1] .= π0           # newborns: zero wealth, ergodic income, good standing
    Λ_stack = zeros(na, nε, ns, N)
    filing_hazard = zeros(N)
    for a in 1:N
        Λ_stack[:, :, :, a] .= dropdims(Λ_cohort; dims = 4)
        # By-age filing hazard: push start-of-age mass through shock then the
        # default choice; the rise in excluded mass is the newly filed, scaled
        # by the good-standing pool it came from.
        shock_s, default_s = comp[a].buffer.stages[1], comp[a].buffer.stages[2]
        good_mass = sum(@view Λ_cohort[:, :, 1, :])
        Λ1 = forward!(shock_s, copy(Λ_cohort))
        Λ2 = forward!(default_s, copy(Λ1))
        new_filers = sum(@view Λ2[:, :, 2, :]) - sum(@view Λ_cohort[:, :, 2, :])
        filing_hazard[a] = good_mass > 0 ? new_filers / good_mass : 0.0
        Λ_cohort = copy(forward!(comp[a], Λ_cohort))
    end

    # Moments — stacked Λ carries the :age axis; integrands read status/wealth #
    #-------------------------------------------------------------------------#
    m = compute_moments(hh, Λ_stack, lmt_env_age(p.peak_age, p))
    mean_assets_xsec   = m.mean_assets / N
    excluded_rate_xsec = m.excluded_rate / N
    excluded_by_age = [sum(@view Λ_stack[:, :, 2, a]) for a in 1:N]      # per-age unit mass ⇒ this is a share

    if verbosity > 0
        @printf "Livshits–MacGee–Tertilt life-cycle bankruptcy (N = %d, σ = %.1f, r = %.3f, β = %.3f)\n" N p.σ p.r p.β
        @printf "  borrowing limit a_min   : %.3f   (garnishment λ = %.2f, readmit ψ = %.2f, stigma χ = %.3f)\n" p.a_min p.λ p.ψ p.χ
        @printf "  per-age cohort mass     : min %.6f, max %.6f (target 1.0)\n" minimum(sum(Λ_stack; dims = (1, 2, 3))) maximum(sum(Λ_stack; dims = (1, 2, 3)))
        @printf "  V finite (a≥0 region)   : %s\n" all(isfinite, @view V_stack[wgrid .>= -1e-9, :, :, :])
        @printf "  mean assets (x-section) : %.4f\n" mean_assets_xsec
        @printf "  excluded share          : %.4f\n" excluded_rate_xsec
        @printf "  filing hazard min/max   : %.4f / %.4f (peak hazard at age %d)\n" minimum(filing_hazard) maximum(filing_hazard) argmax(filing_hazard)
        @printf "  excluded share by age   : age 1 %.4f, peak %.4f (at age %d), final %.4f\n" excluded_by_age[1] maximum(excluded_by_age) argmax(excluded_by_age) excluded_by_age[N]
    end

    return (; V = V_stack, Λ = Λ_stack,
              mean_assets = mean_assets_xsec, excluded_rate = excluded_rate_xsec,
              excluded_by_age, filing_hazard)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Livshits–MacGee–Tertilt life-cycle bankruptcy household…")
    @time lmt_solve()
end
