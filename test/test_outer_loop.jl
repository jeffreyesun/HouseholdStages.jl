using Test
using HouseholdStages
using HouseholdStages: _sup_norm_diff, HhsLiftTag
using ForwardDiff: Dual, value


# A tiny Aiyagari-shaped chain — 2 income states, 20-point wealth grid —
# small enough that VFI converges in tens of iterations and the
# steady-state inner solves are sub-second.

function _tiny_aiyagari_layout(; N_w::Int = 20)
    return GriddedLayout(
        :wealth => GriddedContinuous(0.0, 30.0, N_w; spacing = :log),
        :income => Discrete([0.6, 1.4]),
    )
end

function _tiny_aiyagari_household(layout::GriddedLayout;
                                  β::Float64 = 0.96, σ::Float64 = 1.5)
    P_y = [0.7 0.3; 0.3 0.7]
    income_shock = MarkovStage(layout; axis = :income, transition_matrix = P_y)
    income_receipt = WealthChangeStage(layout;
        wealth_post  = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis         = :wealth,
    )
    u(c) = c < 0 ? -Inf : (σ == 1.0 ? log(c) : (c^(1 - σ)) / (1 - σ))
    savings = ConsumptionSavingsStage(layout;
        β               = β,
        utility         = (cell, c; env) -> u(c),
        axis            = :wealth,
    )
    chain = income_shock ∘ income_receipt ∘ savings
    return define_moments!(chain;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end

function _tiny_aiyagari_prices(K::Real; α::Float64 = 0.36, δ::Float64 = 0.08, L::Float64 = 1.0, A::Real = 1.0)
    r = α * A * (K / L)^(α - 1) - δ
    w = (1 - α) * A * (K / L)^α
    return (; r, w)
end


@testset "solve_vfi_steady_state_given_env! — converges on a tiny chain" begin
    layout = _tiny_aiyagari_layout()
    hh = _tiny_aiyagari_household(layout)
    env = (; K = 5.0, _tiny_aiyagari_prices(5.0)...)

    res = solve_vfi_steady_state_given_env!(hh, env;
                                             tol = 1e-6, maxiter = 1000)
    @test res.converged
    @test res.iters > 1
    @test maximum(abs, backward!(hh, res.V, env) .- res.V) < 1e-5
end


@testset "solve_lambda_steady_state_given_env! — converges to a probability distribution" begin
    layout = _tiny_aiyagari_layout()
    hh = _tiny_aiyagari_household(layout)
    env = (; K = 5.0, _tiny_aiyagari_prices(5.0)...)

    # Seed kernels by running backward at this env.
    solve_vfi_steady_state_given_env!(hh, env;
                                       tol = 1e-6, maxiter = 1000)

    res = solve_lambda_steady_state_given_env!(hh;
                                                tol = 1e-6, maxiter = 50_000)
    @test res.converged
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-8)
    @test all(res.Λ .>= -1e-12)
end


@testset "solve_steady_state_given_env! — bundles V + Λ at one env" begin
    layout = _tiny_aiyagari_layout()
    hh = _tiny_aiyagari_household(layout)
    env = (; K = 5.0, _tiny_aiyagari_prices(5.0)...)

    res = solve_steady_state_given_env!(hh, env;
                                         vfi_tol = 1e-6, lambda_tol = 1e-6)
    @test res.history.vfi_iters > 1
    @test res.history.lambda_iters > 1
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-8)
    # Bellman fixed-point check
    @test maximum(abs, backward!(hh, res.V, env) .- res.V) < 1e-5
    @test res.moments.K_supplied > 0.0
end


@testset "solve_steady_state_given_env! — V_init / Λ_init kwargs warm-start" begin
    layout = _tiny_aiyagari_layout()
    hh = _tiny_aiyagari_household(layout)
    env = (; K = 5.0, _tiny_aiyagari_prices(5.0)...)

    res1 = solve_steady_state_given_env!(hh, env;
                                          vfi_tol = 1e-6, lambda_tol = 1e-6)
    # Re-solve with the previous (V, Λ) as warm-start: should converge in
    # vastly fewer iters because we're already at the fixed point.
    res2 = solve_steady_state_given_env!(hh, env;
                                          V_init = res1.V, Λ_init = res1.Λ,
                                          vfi_tol = 1e-6, lambda_tol = 1e-6)
    @test res2.history.vfi_iters < res1.history.vfi_iters
    @test res2.history.lambda_iters < res1.history.lambda_iters
end


# A block must close #
#--------------------#
# Individual stages regrid freely; the fixed-point solvers feed a block's own output back into its
# input, so the BLOCK's two ends have to agree. Layout equality, not size equality — a same-size
# crosswalk is equally ill-posed and the sup-norm would never notice.

@testset "the fixed-point solvers refuse a block that does not close" begin
    l4  = GriddedLayout(:x => GriddedContinuous([0.0, 1.0, 2.0, 3.0]))
    l2  = GriddedLayout(:x => GriddedContinuous([0.5, 2.5]))
    l4′ = GriddedLayout(:x => GriddedContinuous([0.2, 1.1, 2.2, 3.3]))

    shrinking = MarkovStage(l4, l2; axis = :x, transition_matrix = fill(0.25, 4, 2)) ∘ IdentityStage(l2)
    @test_throws "must be the same layout" solve_vfi_steady_state_given_env!(shrinking, nothing)
    @test_throws "must be the same layout" solve_lambda_steady_state_given_env!(shrinking)

    # Same size at both ends, different coordinates: a shape check would pass this.
    crossing = DeterministicContinuousStage(l4, l4′; axis = :x,
                                            destination = (; x, env) -> 0.5x + 0.3) ∘ IdentityStage(l4′)
    @test_throws "must be the same layout" solve_vfi_steady_state_given_env!(crossing, nothing)
    @test_throws "must be the same layout" solve_transition_given_env_path!(
        crossing.spec, [nothing]; Λ_0 = zeros(4), V_T = zeros(4), boundaries = boundaries(crossing))

    # The square block it is built from still solves.
    square = MarkovStage(l4; axis = :x, transition_matrix = fill(0.25, 4, 4)) ∘ IdentityStage(l4)
    @test solve_vfi_steady_state_given_env!(square, nothing).converged
end


# The shared loop and its convergence metrics #
#---------------------------------------------#

@testset "the maxiter cap errors on the very step that converges" begin
    layout = _tiny_aiyagari_layout()
    hh  = _tiny_aiyagari_household(layout)
    env = (; K = 5.0, _tiny_aiyagari_prices(5.0)...)
    cold() = zero(V_start_buffer(hh))

    n = solve_vfi_steady_state_given_env!(hh, env; V_init = cold(), tol = 1e-6).iters
    # The cap is tested after the increment and without reading the freshly computed `diff`, so a
    # run that converges ON the maxiter-th step still errors; `n + 1` is the smallest cap it passes.
    @test_throws "failed to converge in $n iterations" solve_vfi_steady_state_given_env!(
        hh, env; V_init = cold(), tol = 1e-6, maxiter = n)
    @test solve_vfi_steady_state_given_env!(hh, env; V_init = cold(), tol = 1e-6, maxiter = n + 1).iters == n
end

