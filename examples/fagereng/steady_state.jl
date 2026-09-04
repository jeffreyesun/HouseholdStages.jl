#####################################################################
# Fagereng steady state — return heterogeneity fattens wealth         #
#####################################################################

# Returns, the return-type process, income, and the wage are exogenous, so
# there is no market to clear: the "outer loop" is a single inner V/Λ
# fixed-point solve at the given env. The whole point is that the household
# block is library stages only — see `model.jl`.
#
# The report runs the model TWICE on the same household block, swapping only
# the env's per-type return vector:
#   • HETEROGENEOUS returns (R = [1.00, 1.04]), and
#   • HOMOGENEOUS returns   (R = [1.02, 1.02], same population average),
# and shows that persistent return heterogeneity FATTENS the wealth
# distribution — higher dispersion (CV of wealth), a larger top-10% share,
# and a wide gap in mean wealth between the high- and low-return types.

include("model.jl")

using Printf

"""
Wealth-distribution summary from the stationary `Λ` and the block's cell
array: total mass, mean wealth, the coefficient of variation of wealth (a
dispersion measure), the top-10% wealth share (a fat-tail measure), and mean
wealth conditional on each return type. `Λ` has dims `(wealth, income, rtype)`
in layout order.
"""
function wealth_distribution(hh, Λ)
    cells  = cell_array(end_layout(hh))                 # (N_w, N_income, N_rtype)
    w_vals = getfield.(cells, :wealth)

    mass   = sum(Λ)
    mean_w = sum(w_vals .* Λ) / mass
    mean_w2 = sum(w_vals .^ 2 .* Λ) / mass
    cv     = sqrt(max(mean_w2 - mean_w^2, 0.0)) / mean_w

    # Top-10% wealth share: order cells by wealth descending, accumulate mass
    # to the top decile, sum the wealth they hold over total wealth.
    wv   = vec(w_vals)
    lv   = vec(Λ) ./ mass
    ord  = sortperm(wv; rev = true)
    cum  = 0.0
    held = 0.0
    for i in ord
        lv[i] <= 0 && continue                             # skip empty (often top) grid cells
        take = min(lv[i], 0.10 - cum)
        held += take * wv[i]
        cum  += take
        cum >= 0.10 - 1e-12 && break
    end
    top10_share = held / mean_w

    # Mean wealth conditional on each return type (rtype is the last axis).
    n_rtype     = size(Λ, 3)
    mean_w_by_t = [sum(@view(w_vals[:, :, t]) .* @view(Λ[:, :, t])) /
                   max(sum(@view Λ[:, :, t]), eps()) for t in 1:n_rtype]

    return (; mass, mean_w, cv, top10_share, mean_w_by_t)
end

"""
Solve the Fagereng household at one return calibration and return the
stationary `(V, Λ)`, the `mean_wealth` moment, and the wealth-distribution
summary. `R_by_type` selects the calibration (heterogeneous vs homogeneous).
"""
function fagereng_solve(p = fagereng_params; R_by_type = p.R_hetero, verbosity = 0)
    hh  = fagereng_household(p)
    env = fagereng_env(p; R_by_type)
    res = solve_steady_state_given_env!(hh, env)
    m   = compute_moments(hh, res.Λ, env)
    d   = wealth_distribution(hh, res.Λ)
    if verbosity > 0
        @printf "  mass(Λ) = %.6f, mean wealth = %.4f, VFI %d / Λ %d\n" d.mass m.mean_w res.history.vfi_iters res.history.lambda_iters
    end
    return (; V = res.V, Λ = res.Λ, mean_wealth = m.mean_wealth, dist = d, history = res.history)
end

"""
Run the heterogeneous-vs-homogeneous return comparison and print the
fattening result: dispersion (CV), top-10% share, and per-type mean wealth.
"""
function fagereng_steady_state(p = fagereng_params; verbosity = 1)
    het = fagereng_solve(p; R_by_type = p.R_hetero)
    hom = fagereng_solve(p; R_by_type = p.R_homog)

    if verbosity > 0
        @printf "Fagereng return heterogeneity (β = %.2f, σ = %.1f)\n" p.β p.σ
        @printf "  return types: persistence diag(P) = %.2f, stationary mass = [0.50, 0.50]\n" p.P_rtype[1, 1]
        @printf "\n  %-26s %12s %12s\n" "" "HETERO" "HOMOG"
        @printf "  %-26s %12s %12s\n" "  gross returns R(rtype)" string(p.R_hetero) string(p.R_homog)
        @printf "  %-26s %12.4f %12.4f\n" "  mean wealth"        het.dist.mean_w       hom.dist.mean_w
        @printf "  %-26s %12.4f %12.4f\n" "  CV of wealth"       het.dist.cv           hom.dist.cv
        @printf "  %-26s %12.4f %12.4f\n" "  top-10% share"      het.dist.top10_share  hom.dist.top10_share
        @printf "\n  per-type mean wealth (hetero): low-R = %.3f, high-R = %.3f  (ratio %.1f×)\n" het.dist.mean_w_by_t[1] het.dist.mean_w_by_t[2] (het.dist.mean_w_by_t[2] / max(het.dist.mean_w_by_t[1], eps()))
        @printf "  per-type mean wealth (homog):  low   = %.3f, high   = %.3f\n" hom.dist.mean_w_by_t[1] hom.dist.mean_w_by_t[2]
        fatter = het.dist.cv > hom.dist.cv && het.dist.top10_share > hom.dist.top10_share
        @printf "\n  ⇒ return heterogeneity FATTENS the wealth distribution: %s\n" fatter
        @printf "    (CV %.1f%% higher, top-10%% share %.1f pp higher)\n" 100 * (het.dist.cv / hom.dist.cv - 1) 100 * (het.dist.top10_share - hom.dist.top10_share)
    end
    return (; het, hom)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Fagereng return-heterogeneity steady state (hetero vs homog)…")
    @time fagereng_steady_state()
end
