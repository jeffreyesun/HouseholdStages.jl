###################################################################
# Arellano steady state — bond-price fixed point ⇄ household block #
###################################################################

# Unlike `examples/default` (exogenous risk-free bond ⇒ a single inner solve),
# the bond price `q(a', y)` here is ENDOGENOUS: it equals the lenders' break-even
# price given next period's default policy, which itself depends on `q`. So the
# outer loop is a damped fixed point on the whole `q` SCHEDULE (an `(N_a, N_y)`
# matrix) — the same shape of tatonnement `examples/aiyagari` runs on a scalar
# `K`, lifted to a schedule:
#
#   guess q → solve the household V/Λ at env (; λ, rf, q) → read the default
#   policy → reprice q'(a',y) = (1/(1+rf))·(1 − Σ_{y'} P[y,y']·default(a',y'))
#   → damp q ← q + speed·(q' − q) → repeat to a fixed point.
#
# Reading the per-cell repay/default choice off the solved `DefaultStage`
# (a library `ArgmaxStage` on `:status`) is ordinary driver work — exactly the
# kind of "reach into the solved block from outside" the framework intends.
# The within-period block is library stages only; see `model.jl`.

include("model.jl")

using Printf
using LinearAlgebra: dot


# Read the default policy off the solved block #
#----------------------------------------------#

"""
The good-standing default policy of the solved chain as a `(N_a, N_y)` `Bool` matrix
`default[ia', iy']` = true iff a good-standing agent entering with assets `a'` and income `y'`
chooses to default. Found by locating the unique `:status`-axis `ArgmaxStage` (the `DefaultStage`)
and reading its solved policy (the chosen status index per cell); index 2 on the good-standing
(`status = 1`) slice is the default choice.
"""
function default_policy(built)
    stages  = collect(built.hh.buffer.stages)
    dstage  = only(filter(s -> s isa ArgmaxStage && s.spec.axis == :status, stages))
    pol     = policy(dstage)                       # (N_a, N_y, N_status): chosen after-status index
    return pol[:, :, 1] .== 2                       # default iff the good-standing cell chose status 2
end


# Reprice the bond off the default policy #
#-----------------------------------------#

"""
The lenders' break-even price schedule `q'(a', y)` given the good-standing default policy
`defmat[ia', iy']` and the income transition `P[y, y']`:
`q'(a',y) = (1/(1+rf))·(1 − Σ_{y'} P[y,y']·default(a',y'))`. Saving (`a' ≥ 0`) is priced risk-free.
"""
function reprice(defmat::AbstractMatrix{Bool}, built, p)
    rf    = p.rf
    ED    = Float64.(defmat) * transpose(built.P_y)          # ED[ia', iy] = E[default | a', current y]
    q     = (1 / (1 + rf)) .* (1 .- ED)
    @inbounds for (ia, a_next) in enumerate(built.grid)
        a_next ≥ 0 && (q[ia, :] .= 1 / (1 + rf))             # no default risk on positive assets
    end
    return q
end


# Outer fixed point on the price schedule #
#-----------------------------------------#

"""
Solve the Arellano steady state by a damped fixed point on the bond-price schedule `q`. At each pass:
solve the household V/Λ at the current `q`, read the default policy, reprice, and nudge
`q ← q + update_speed·(q' − q)`. Returns the converged `q`, `V`, `Λ`, the default policy, and the
endowment grid / transition (for reporting). `q` starts risk-free everywhere.
"""
function arellano_steady_state(p = arellano_params;
                               update_speed = 0.5,
                               tol          = 1e-5,
                               max_iter     = 200,
                               verbosity    = 1)
    built       = arellano_household(p)
    hh          = built.hh
    N_a, N_y    = p.N_a, p.N_y

    q = fill(1 / (1 + p.rf), N_a, N_y)              # initial guess: risk-free bond everywhere
    V, Λ = nothing, nothing
    q_err = Inf
    iterations = 0
    defmat = falses(N_a, N_y)

    while iterations < max_iter
        env = arellano_env(q, p)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        (; V, Λ) = res

        defmat = default_policy(built)
        q_new  = reprice(defmat, built, p)
        q_err  = maximum(abs.(q_new .- q))
        iterations += 1

        verbosity > 1 && @printf "  iter %3d: q_err = %.3e\n" iterations q_err
        q_err <= tol && break
        q .+= update_speed .* (q_new .- q)
    end

    converged = q_err <= tol
    converged || @warn "Arellano q fixed point stuck at tol $tol after $iterations iterations (q_err = $q_err)."

    return (; q, V, Λ, defmat, built, p, iterations, converged, q_err)
end


# Reporting helpers #
#-------------------#

"""
The annualized credit spread schedule `spread(a',y) = (1+rf)/((1+rf)·q) − 1 = 1/q·... ` — concretely
the one-period yield gap `(1/q − (1+rf))` (zero when risk-free). Returned as an `(N_a, N_y)` matrix in
basis points (×10⁴), with a large cap where `q → 0` (debt the lenders refuse).
"""
function spread_bps(q::AbstractMatrix, p)
    rf = p.rf
    s  = similar(q)
    @inbounds for i in eachindex(q)
        s[i] = q[i] > 1e-8 ? (1 / q[i] - (1 + rf)) * 1e4 : Inf
    end
    return s
end

"""
The stationary on-path default rate: the mass of good-standing agents who default NEXT period, over
total good-standing mass. The end-of-period `Λ` carries this period's income, but the default
decision follows next period's income shock, so the per-cell default probability is the
income-convolved `ED = default·Pᵀ` (the very object lenders price off) — not `defmat` directly.
"""
function stationary_default_rate(res)
    Λ_good = res.Λ[:, :, 1]
    good   = sum(Λ_good)
    good <= 0 && return 0.0
    ED = Float64.(res.defmat) * transpose(res.built.P_y)   # ED[a', y] = E[default next period | a', y]
    return sum(Λ_good .* ED) / good
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Arellano (2008) sovereign-default steady state…")
    @time res = arellano_steady_state(; verbosity = 2)
    p = res.p

    m = compute_moments(res.built.hh, res.Λ, arellano_env(res.q, p))

    println(res.converged ? "q fixed point converged in $(res.iterations) iterations." :
                            "q fixed point DID NOT CONVERGE in $(res.iterations) iterations (q_err = $(res.q_err)).")
    @printf "  ΣΛ              = %.6f\n" sum(res.Λ)
    @printf "  mean assets     = %+.4f   (negative ⇒ net debtor)\n" m.mean_assets
    @printf "  excluded rate   = %.4f\n" m.excluded_rate
    @printf "  default rate    = %.4f%%  (per period, among good standing)\n" 100 * stationary_default_rate(res)

    # Spreads rising with debt, at the median income state.
    s   = spread_bps(res.q, p)
    iy  = (p.N_y + 1) ÷ 2
    g   = res.built.grid
    println("\n  Spread schedule at median income (y = $(round(res.built.y_grid[iy]; digits=3))):")
    @printf "    %10s %12s %10s\n" "a'" "q" "spread(bps)"
    for a_target in (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1)
        ia = argmin(abs.(g .- a_target))
        sp = isfinite(s[ia, iy]) ? @sprintf("%10.1f", s[ia, iy]) : "       Inf"
        @printf "    %10.3f %12.4f %s\n" g[ia] res.q[ia, iy] sp
    end

    # Default region: lowest income vs highest income (countercyclical default).
    @printf "\n  Default frontier (least debt a' that still defaults next period), by income:\n"
    for iy2 in 1:p.N_y
        col = res.defmat[:, iy2]
        thr = findlast(col)
        label = thr === nothing ? "  none" : @sprintf("%+.3f", g[thr])
        @printf "    y = %.3f :  a' frontier = %s\n" res.built.y_grid[iy2] label
    end
end
