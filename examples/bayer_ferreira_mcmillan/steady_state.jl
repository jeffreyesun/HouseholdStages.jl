######################################################################
# BFM neighborhood sorting — price-clearing equilibrium (outer loop)  #
######################################################################

# The household block (neighborhood choice) is exercised inside an OUTER
# loop that closes the SORTING EQUILIBRIUM: prices adjust so each
# neighborhood's population equals its fixed housing capacity, and the
# rich-share composition is updated to its realized value each pass.
# Richer types are less price-sensitive (price disutility ∝ 1/income), so
# they outbid the poor for high-amenity neighborhoods — the BFM income
# sorting. The per-price inner V/Λ solve is delegated to
# `solve_steady_state_given_env!`; the price/composition tatonnement is here.

include("model.jl")

using Printf

"""
Solve the BFM sorting equilibrium by tatonnement: raise a neighborhood's
price when its population exceeds capacity, lower it otherwise; update the
rich-share composition to the realized value; repeat until prices clear.
"""
function bfm_steady_state(p = params;
                          price_speed = 0.12,
                          comp_damping = 0.3,
                          tol = 1e-3,
                          maxiter = 600,
                          verbosity = 1)
    hh = bfm_household(p)
    nN = length(p.neighborhoods)

    price      = zeros(nN)
    rich_share = fill(0.5, nN)
    V, Λ = nothing, nothing
    moments = nothing
    iters = 0
    converged = false

    while iters < maxiter
        env = bfm_env(price, rich_share, p)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        (; V, Λ) = res
        moments = compute_moments(hh, Λ, env)

        pop  = [moments.pop_south, moments.pop_midtown, moments.pop_heights]
        rich = [moments.rich_south, moments.rich_midtown, moments.rich_heights]
        rich_share_new = rich ./ max.(pop, eps())

        excess = pop .- p.capacity          # excess demand per neighborhood
        gap = maximum(abs.(excess))
        iters += 1
        verbosity > 0 && (iters <= 4 || iters % 25 == 0) &&
            @printf "  iter %d: max|excess demand| = %.5f, price = [%.3f, %.3f, %.3f]\n" iters gap price...

        if gap < tol
            converged = true
            rich_share = rich_share_new
            break
        end
        # Raise price where demand exceeds capacity (normalize: south price ≡ 0 numeraire).
        price .+= price_speed .* excess
        price .-= price[1]
        rich_share .= (1 - comp_damping) .* rich_share .+ comp_damping .* rich_share_new
    end

    converged || @warn "BFM price tatonnement not converged in $iters iterations."
    return (; price, rich_share, V, Λ, moments, iters, converged)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving BFM neighborhood-sorting equilibrium…")
    @time res = bfm_steady_state()
    m = res.moments
    println(res.converged ? "Sorting equilibrium converged in $(res.iters) iterations." :
                            "DID NOT CONVERGE in $(res.iters) iterations.")
    pop  = [m.pop_south, m.pop_midtown, m.pop_heights]
    rich = [m.rich_south, m.rich_midtown, m.rich_heights]
    println("  Clearing prices (south ≡ 0): [$(join(round.(res.price; digits=3), ", "))]")
    @printf "  Population by neighborhood: south=%.3f midtown=%.3f heights=%.3f (capacity 1/3 each)\n" pop...
    @printf "  Rich share by neighborhood: south=%.3f midtown=%.3f heights=%.3f\n" (rich ./ max.(pop, eps()))...
    println("  (income sorting ⇔ rich share rises with amenity: south < midtown < heights)")
    @printf "  ΣΛ = %.6f\n" sum(res.Λ)
end
