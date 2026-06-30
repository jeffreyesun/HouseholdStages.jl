###############################################################
# De Nardi solve — finite-horizon sweep + dynastic closure    #
###############################################################

# Two example-side outer loops over the `replicate_age` block (`model.jl`):
#
#   (1) FINITE-HORIZON BACKWARD SWEEP `a = N…1`. A `ProductStage`'s `backward!`
#       runs each age independently, so we thread each age-(a+1)'s continuation
#       value into age-a's component by hand, seating every age's savings policy.
#       Done once — policies do not depend on the dynastic distribution.
#
#   (2) DYNASTIC FIXED POINT. The dynasty links the assets the dying bequeath to
#       the assets newborns inherit. We iterate the newborn inherited-wealth
#       distribution `g`: seed a unit cohort with `g` (via an `EntryStage`), push
#       it forward `a = 1…N` (each age the exit composite leaks the dying
#       fraction), tally the cross-age distribution of bequeathed assets, and set
#       the next `g` to that distribution. At the fixed point one death funds one
#       newborn (total bequest mass = total birth mass = 1), so the dynasty is a
#       genuine cross-generation stationary distribution.
#
# Each age's deaths leave their POST-SAVINGS assets `a'`. Since survival `s(a)` is
# scalar at each age, exit scales the cohort uniformly: the post-savings mass is
# `Λ_next / s(a)` (`Λ_next` = the survivors carried into age a+1), and the dying
# fraction is `(1−s(a))` of it. The final age-N cohort all die (no age N+1), so its
# whole post-savings mass is bequeathed — the bequest masses then telescope to 1.

include("model.jl")

using Printf

"""
Share of total wealth held by the richest `q` fraction of the population, given
per-wealth-grid population masses `mass_w` (ascending grid `wgrid`). Walks the grid
from the top, accumulating population until the top-`q` mass is reached (splitting
the marginal grid cell), and returns that group's wealth share.
"""
function dn_top_share(wgrid::AbstractVector, mass_w::AbstractVector, q::Real)
    pop   = sum(mass_w)
    total = sum(wgrid .* mass_w)
    (pop ≤ 0 || total ≤ 0) && return NaN
    target = q * pop
    acc_pop = 0.0; acc_wealth = 0.0
    for i in length(wgrid):-1:1
        mass_w[i] ≤ 0 && continue                       # skip empty cells, do NOT stop
        take = min(mass_w[i], target - acc_pop)
        acc_pop    += take
        acc_wealth += take * wgrid[i]
        acc_pop ≥ target - 1e-12 && break               # the top-q population is filled
    end
    return acc_wealth / total
end

"""
Solve the De Nardi dynastic life-cycle model: a single finite-horizon backward
sweep to seat policies, then an outer fixed point on the newborn inherited-wealth
distribution `g` (the dynastic link). Each dynastic pass seeds newborns with `g`
through an `EntryStage`, runs the forward cohort sweep accumulating bequeathed
assets, and updates `g`. Returns the seated value, the per-age asset profile and
surviving-mass profile, the pooled cross-sectional wealth distribution, and wealth
moments (mean, top-decile share, total pooled mass).
"""
function de_nardi_solve(p = de_nardi_params; max_outer = 60, tol = 1e-7, verbosity = 1)
    hh      = de_nardi_household(p)
    product = hh.buffer.stages[1]
    comp    = product.buffer.components
    comp_layout = input_layout(comp[1])
    out_layout  = product.buffer.output_layout
    nw, nε, N = p.N_w, length(p.ε_grid), p.N

    env_age(a) = (; r = p.r, y = dn_age_earnings(a, p), age = a)
    wgrid = getproperty.(cell_array(comp_layout)[:, 1, 1, 1], :wealth)   # ascending wealth grid

    # (1) Backward induction — V_{N+1} = 0; warm-glow bequest is inside the exit stage #
    #--------------------------------------------------------------------------------#
    V_next = zeros(nw, nε, 1, 1)
    for a in N:-1:1
        V_next = copy(backward!(comp[a], V_next, env_age(a)))
    end
    V_finite = all(isfinite, V_next)

    # (2) Dynastic fixed point on the newborn inherited-wealth distribution `g` #
    #-------------------------------------------------------------------------#
    π0  = dn_income_stationary(p)
    g_w = zeros(nw); g_w[1] = 1.0                       # initial guess: newborns start with no inheritance

    Λ_stack       = zeros(nw, nε, N)                    # start-of-age distribution per age
    mass_by_age   = zeros(N)
    age_mean_wlth = zeros(N)
    dyn_gap = NaN
    iters_done = 0
    for outer in 1:max_outer
        iters_done = outer
        # Seed newborns with the inherited-wealth distribution `g` ⊗ ergodic income,
        # injected through an EntryStage (the dynastic link's `g`).
        g_field = zeros(nw, nε, 1, 1)
        @views for j in 1:nε
            g_field[:, j, 1, 1] .= g_w .* π0[j]
        end
        entry    = EntryStage(comp_layout; entry = g_field)
        backward!(entry, zeros(nw, nε, 1, 1), NamedTuple())          # seat g
        Λ_enter  = copy(forward!(entry, zeros(nw, nε, 1, 1)))        # newborns = g_field

        bequest_w = zeros(nw)
        for a in 1:N
            Λ_stack[:, :, a] .= dropdims(Λ_enter; dims = (3, 4))
            mass_by_age[a]    = sum(Λ_enter)
            age_mean_wlth[a]  = mass_by_age[a] > 0 ?
                sum(wgrid .* vec(sum(Λ_stack[:, :, a]; dims = 2))) / mass_by_age[a] : 0.0
            Λ_next = copy(forward!(comp[a], Λ_enter))
            s_a    = dn_survival(a, p)
            post_w = vec(sum(dropdims(Λ_next ./ s_a; dims = (3, 4)); dims = 2))  # post-savings wealth marginal
            frac   = a < N ? (1 - s_a) : 1.0                          # dying fraction (all of the final cohort)
            bequest_w .+= frac .* post_w
            Λ_enter = Λ_next
        end

        g_new   = bequest_w ./ sum(bequest_w)
        dyn_gap = maximum(abs, g_new .- g_w)
        g_w     = g_new
        dyn_gap < tol && break
    end

    # Moments — pooled cross-section over living ages (mass shrinks with age) #
    #-----------------------------------------------------------------------#
    pooled_w   = vec(sum(Λ_stack; dims = (2, 3)))       # wealth marginal pooled over income & age
    total_mass = sum(pooled_w)
    mean_wealth = sum(wgrid .* pooled_w) / total_mass
    top10 = dn_top_share(wgrid, pooled_w, 0.10)
    top1  = dn_top_share(wgrid, pooled_w, 0.01)

    if verbosity > 0
        @printf "De Nardi dynastic life cycle (N = %d, σ = %.1f, r = %.3f, β = %.3f, φ = %.1f, κ = %.1f)\n" N p.σ p.r p.β p.φ p.κ
        @printf "  dynastic closure        : %d iters, final |Δg| = %.2e\n" iters_done dyn_gap
        @printf "  V finite                : %s\n" V_finite
        @printf "  newborn mass / total pop: %.4f / %.4f\n" mass_by_age[1] total_mass
        @printf "  surviving mass age 1→N  : %.4f → %.4f\n" mass_by_age[1] mass_by_age[N]
        @printf "  mean wealth (pooled)    : %.4f\n" mean_wealth
        @printf "  asset profile 1/peak/N  : %.3f / %.3f (age %d) / %.3f\n" age_mean_wlth[1] maximum(age_mean_wlth) argmax(age_mean_wlth) age_mean_wlth[N]
        @printf "  top 10%% wealth share    : %.4f\n" top10
        @printf "  top 1%%  wealth share    : %.4f\n" top1
    end

    return (; V = V_next, Λ = Λ_stack, mean_wealth, top10, top1,
              total_mass, mass_by_age, age_mean_wealth = age_mean_wlth,
              g = g_w, dyn_gap, iters = iters_done)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving De Nardi dynastic life-cycle household…")
    @time de_nardi_solve()
end
