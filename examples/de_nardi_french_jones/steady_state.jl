###############################################################
# DFJ solve — finite-horizon backward + shrinking-cohort sweep #
###############################################################

# A retired cohort has no stationary steady state: the solve is a single backward
# sweep over ages (`a = N…1`) followed by a forward cohort simulation (`a = 1…N`).
# Both are example-side outer-loop logic over the `replicate_age` block (`model.jl`).
# The `ProductStage`'s `backward!` runs each age independently, so the driver
# threads each age-(a+1)'s continuation value into age-a's component by hand,
# feeding each age its own `env` (age `a`, rate `r`, pension). On the forward pass
# the exit composite leaks the dying fraction at every age, so the COHORT SHRINKS
# with age — that is correct here (a retired cohort dies out; there is no entry).
#
# The headline objects: mean wealth among survivors by age (the slow-dissaving
# asset profile that the medical-expense risk produces) and the surviving-mass
# profile (the cohort's mortality curve).

include("model.jl")

using Printf

"""
Solve the DFJ retired-cohort life-cycle model by backward induction + a forward
cohort simulation in which mass leaks to mortality each age. Newborns (newly
retired) start at assets `p.w_init` with the stationary health distribution.
Returns the seated value, the per-age survivor mean-wealth profile, the
surviving-mass (mortality) profile, and the stacked distribution.
"""
function dfj_solve(p = dfj_params; verbosity = 1)
    hh      = dfj_household(p)
    product = hh.buffer.stages[1]
    comp    = product.buffer.components
    comp_layout = input_layout(comp[1])
    nw, nh, N = p.N_w, size(p.P_h, 1), p.N

    env_age(a) = (; r = p.r, age = a, pension = p.pension)
    wgrid = getproperty.(cell_array(comp_layout)[:, 1, 1, 1], :wealth)

    # Backward induction — V_{N+1} = 0; warm-glow bequest is inside the exit stage #
    #----------------------------------------------------------------------------#
    V_next = zeros(nw, nh, 1, 1)
    for a in N:-1:1
        V_next = copy(backward!(comp[a], V_next, env_age(a)))
    end
    V_finite = all(isfinite, V_next)

    # Forward cohort simulation — newly-retired at age 1, mass shrinks with age #
    #-------------------------------------------------------------------------#
    π0   = dfj_health_stationary(p)
    iw   = argmin(abs.(wgrid .- p.w_init))         # nearest grid node to the initial assets
    Λ_enter = zeros(nw, nh, 1, 1)
    @views for j in 1:nh
        Λ_enter[iw, j, 1, 1] = π0[j]
    end

    Λ_stack       = zeros(nw, nh, N)
    mass_by_age   = zeros(N)
    age_mean_wlth = zeros(N)
    for a in 1:N
        Λ_stack[:, :, a] .= dropdims(Λ_enter; dims = (3, 4))
        mass_by_age[a]    = sum(Λ_enter)
        age_mean_wlth[a]  = mass_by_age[a] > 0 ?
            sum(wgrid .* vec(sum(Λ_stack[:, :, a]; dims = 2))) / mass_by_age[a] : 0.0
        Λ_enter = copy(forward!(comp[a], Λ_enter))
    end

    if verbosity > 0
        @printf "De Nardi–French–Jones retired cohort (N = %d, σ = %.1f, r = %.3f, β = %.3f)\n" N p.σ p.r p.β
        @printf "  V finite                : %s\n" V_finite
        @printf "  surviving mass age 1→N  : %.4f → %.4f\n" mass_by_age[1] mass_by_age[N]
        @printf "  mean wealth age 1→N     : %.4f → %.4f\n" age_mean_wlth[1] age_mean_wlth[N]
        @printf "  asset half-life reached : age %d (wealth first ≤ ½·initial)\n" something(findfirst(<=(0.5 * age_mean_wlth[1]), age_mean_wlth), N)
        @printf "  mean medical (good/bad) at age %d: %.3f / %.3f\n" N dfj_medical(1, N, p) dfj_medical(2, N, p)
        @printf "  survival (good/bad) at age %d   : %.3f / %.3f\n" N dfj_survival(1, N, p) dfj_survival(2, N, p)
        println("  per-age survivor wealth profile:")
        for a in 1:N
            @printf "    age %2d : mass %.4f   mean wealth %.4f\n" a mass_by_age[a] age_mean_wlth[a]
        end
    end

    return (; V = V_next, Λ = Λ_stack, mass_by_age,
              age_mean_wealth = age_mean_wlth, V_finite)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving De Nardi–French–Jones retired cohort…")
    @time dfj_solve()
end
