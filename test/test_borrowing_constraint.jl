using Test
using HouseholdStages

@testset "BorrowingConstraintStage — array mask sets V to -Inf on infeasible cells" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([-1.0, 0.0, 1.0, 2.0]),
        :y => Discrete([0.5, 1.0]),
    )
    mask = falses(4, 2)
    mask[1, :] .= true   # wealth = -1 infeasible
    stage = BorrowingConstraintStage(layout; infeasible = mask)

    V_end = reshape(Float64.(1:8), (4, 2))
    V_start = backward!(stage, V_end, NamedTuple())

    @test V_start[1, 1] == -Inf
    @test V_start[1, 2] == -Inf
    @test V_start[2:end, :] == V_end[2:end, :]
end

@testset "BorrowingConstraintStage — forward is identity on Λ" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :z => Discrete([1, 2]),
    )
    mask = falses(3, 2); mask[1, :] .= true
    stage = BorrowingConstraintStage(layout; infeasible = mask)

    Λ_start = rand(3, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start)
    @test Λ_end == Λ_start
    @test Λ_end !== Λ_start
    @test sum(Λ_end) ≈ sum(Λ_start)
end

@testset "BorrowingConstraintStage — closure form materialises mask from env" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([-1.0, 0.0, 1.0, 2.0]),
        :y => Discrete([0.5, 1.0]),
    )
    # State + env-dependent constraint: infeasible if wealth < env.w_min.
    # env is passed directly to the closure (no Ref wrapping).
    stage = BorrowingConstraintStage(layout;
        infeasible = (; wealth, env) -> wealth < env.w_min,
    )

    V_end = reshape(Float64.(1:8), (4, 2))

    # With w_min = 0: wealth = -1 is infeasible.
    V_start1 = copy(backward!(stage, V_end, (; w_min = 0.0)))
    @test V_start1[1, :] == [-Inf, -Inf]
    @test V_start1[2:end, :] == V_end[2:end, :]

    # With w_min = 0.5: wealth ∈ {-1, 0} infeasible.
    V_start2 = copy(backward!(stage, V_end, (; w_min = 0.5)))
    @test V_start2[1, :] == [-Inf, -Inf]
    @test V_start2[2, :] == [-Inf, -Inf]
    @test V_start2[3:end, :] == V_end[3:end, :]
end

@testset "BorrowingConstraintStage — closure and array forms agree" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([-1.0, 0.0, 1.0, 2.0, 3.0]),
        :y => Discrete([0.5, 1.0, 1.5]),
    )
    mask = [w < 0.0 for w in [-1.0, 0.0, 1.0, 2.0, 3.0], y in [0.5, 1.0, 1.5]]
    stage_arr = BorrowingConstraintStage(layout; infeasible = mask)
    stage_fn  = BorrowingConstraintStage(layout;
        infeasible = (; wealth) -> wealth < 0.0,
    )

    V_end = randn(5, 3)
    V_arr = copy(backward!(stage_arr, V_end, NamedTuple()))
    V_fn  = copy(backward!(stage_fn,  V_end, NamedTuple()))
    @test V_arr == V_fn
end

@testset "BorrowingConstraintStage — duality identity (excluding -Inf cells)" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :y => Discrete([0.5, 1.0]),
    )
    mask = falses(4, 2); mask[1, :] .= true
    stage = BorrowingConstraintStage(layout; infeasible = mask)

    V_end = randn(4, 2)
    Λ = rand(4, 2); Λ[1, :] .= 0.0
    Λ ./= sum(Λ)

    V_start = backward!(stage, V_end, NamedTuple())
    Λ_end   = forward!(stage, Λ)
    V_start_safe = ifelse.(isfinite.(V_start), V_start, 0.0)
    @test isapprox(sum(V_start_safe .* Λ), sum(V_end .* Λ_end); atol = 1e-12)
end

@testset "BorrowingConstraintStage — composition with MarkovStage" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(
        :wealth => GriddedContinuous([-1.0, 0.0, 1.0, 2.0]),
        :z => Discrete([0.5, 1.5]),
    )
    markov = MarkovStage(layout; axis = :z, transition_matrix = P)
    mask   = falses(4, 2); mask[1, :] .= true
    bc     = BorrowingConstraintStage(layout; infeasible = mask)

    chain = markov ∘ bc
    V_end = ones(4, 2)
    V_start = backward!(chain, V_end, NamedTuple())
    @test all(V_start[1, :] .== -Inf)
    @test all(isapprox.(V_start[2:end, :], 1.0; atol = 1e-12))
end

@testset "BorrowingConstraintStage — is a UtilityStage with no static env deps" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0]),
        :y => Discrete([0.5, 1.0]),
    )
    stage = BorrowingConstraintStage(layout; infeasible = falses(2, 2))
    @test stage isa UtilityStage
    @test static_env_deps(typeof(stage.spec)) === NamedTuple()
    @test effective_env_slice(stage) === ()
end

@testset "BorrowingConstraintStage — shape check on the array form" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :y => Discrete([0.5, 1.0]),
    )
    @test_throws AssertionError BorrowingConstraintStage(layout; infeasible = falses(4, 3))
end

@testset "_sup_norm_diff — skips non-finite differences (VFI convergence-norm fix)" begin
    @test HouseholdStages._sup_norm_diff([1.0, -Inf, 3.0], [0.0, -Inf, 1.0]) == 2.0   # was NaN
    @test HouseholdStages._sup_norm_diff([1.0, 2.0], [0.0, 0.0]) == 2.0               # all-finite unchanged
    @test HouseholdStages._sup_norm_diff([-Inf, -Inf], [-Inf, -Inf]) == 0.0           # all-masked → 0
end

@testset "BorrowingConstraintStage — full VFI converges with a feasibility mask (regression)" begin
    # Regression for the silent false-convergence bug: the constraint's `-Inf` cells made the
    # sup-norm `maximum(abs, V_new .- V)` evaluate `-Inf - (-Inf) = NaN`, so `while NaN > tol`
    # was false → VFI exited after ~2 passes returning converged=true. It must really iterate now.
    P = [0.7 0.2 0.1; 0.2 0.6 0.2; 0.1 0.2 0.7]
    layout = GriddedLayout(:wealth => GriddedContinuous(0.0, 60.0, 60; spacing = :log),
                           :income => Discrete([0.6, 1.0, 1.4]))
    shock   = MarkovStage(layout; axis = :income, transition_matrix = P)
    receipt = IncomeStage(layout)
    bc      = BorrowingConstraintStage(layout; infeasible = (; wealth) -> wealth < 2.0)
    savings = ConsumptionSavingsStage(layout; β = 0.96, utility = (cell, c; env) -> u_crra(c, Val(1.5)))
    hh  = bc ∘ shock ∘ receipt ∘ savings
    res = solve_steady_state_given_env!(hh, (; r = 0.03, w = 1.0))
    @test res.history.vfi_iters > 10        # really iterated (not the ~2-pass silent exit)
    @test count(==(-Inf), res.V) > 0        # the mask is present
    @test count(isfinite, res.V) > 0        # feasible region is finite
    @test isfinite(sum(res.Λ)) && sum(res.Λ) > 0
end
