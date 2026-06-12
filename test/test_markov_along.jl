using Test
using HouseholdStages

@testset "MarkovStage — construction and field checks" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :income, transition_matrix = P)
    @test stage isa MarkovStage
    @test stage.spec.axis === :income
    @test stage.spec.transition_matrix === P
    @test input_layout(stage) === layout
    @test output_layout(stage) === layout
    @test size(stage.scratch.V_start) == (4, 2)
    @test size(stage.scratch.Λ_end)   == (4, 2)
end

@testset "MarkovStage — backward & forward correctness" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :income, transition_matrix = P)

    V_end = ones(4, 2)
    V_start = backward!(stage, V_end, nothing)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))

    V_end2 = zeros(4, 2); V_end2[:, 1] .= 1.0; V_end2[:, 2] .= 4.0
    V_start2 = backward!(stage, V_end2, nothing)
    @test all(isapprox.(V_start2[:, 1], 0.7*1.0 + 0.3*4.0; atol = 1e-12))
    @test all(isapprox.(V_start2[:, 2], 0.3*1.0 + 0.7*4.0; atol = 1e-12))

    Λ_start = rand(4, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start)
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
end

@testset "MarkovStage — duality identity" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 0.5, 1.0])),
        StateAxis(:z,      discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :z, transition_matrix = P)

    V_out = randn(3, 2)
    Λ_in  = rand(3, 2); Λ_in ./= sum(Λ_in)

    V_in  = backward!(stage, V_out, nothing)
    Λ_out = forward!(stage, Λ_in)

    # For a pure-Markov stage the flow payoff `r` is zero, so duality
    # reduces to ⟨V_in, Λ_in⟩ ≈ ⟨V_out, Λ_out⟩.
    @test isapprox(sum(V_in .* Λ_in), sum(V_out .* Λ_out); atol = 1e-12)
end

@testset "MarkovStage — backward axis=1 (first dim)" begin
    # axis_dim == 1 hits the no-permute fast path in the transition contraction.
    P = [0.9 0.1; 0.2 0.8]
    layout = GriddedLayout(
        StateAxis(:z, discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = MarkovStage(layout; axis = :z, transition_matrix = P)
    V_end = ones(2, 3)
    V_start = backward!(stage, V_end, nothing)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))
end

@testset "MarkovStage — static_env_deps is empty" begin
    @test static_env_deps(HouseholdStages.MarkovStageSpec) === NamedTuple()
end

@testset "MarkovStage — type stability" begin
    P = [0.9 0.1; 0.2 0.8]
    layout = GriddedLayout(
        StateAxis(:z, discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = MarkovStage(layout; axis = :z, transition_matrix = P)
    V_end = ones(2, 3)
    @inferred backward!(stage, V_end, nothing)
end

# Dep-varying transition: an :addiction Markov whose into-addiction probability
# rises with unemployment. addiction levels [0,1] = clean/addicted; employment
# levels [1,2] = employed/unemployed. Per stratum the ROW-stochastic matrix is
# T[from,to] = [[1-pin pin]; [pout 1-pout]] with pin (clean→addicted) higher when
# unemployed. The stored fiber is K = Tᵀ, varying along :employment only — a closure
# `(; employment) -> T` reading no env, so it is seated once at allocation.
@testset "MarkovStage — dep-varying transition (matrix closure)" begin
    layout = GriddedLayout(
        StateAxis(:addiction,  discrete_finite([0, 1])),
        StateAxis(:employment, discrete_finite([1, 2])),
    )
    pin(emp)  = emp == 2 ? 0.30 : 0.05        # into-addiction prob — higher unemployed
    pout      = 0.20                          # recovery prob (employment-independent)
    Tmat(emp) = [1-pin(emp)  pin(emp); pout  1-pout]

    stage = MarkovStage(layout; axis = :addiction,
                        transition_matrix = (; employment) -> Tmat(employment))

    # backward! seats the kernel; its compact storage is dep-only (2,2,employment) and each
    # fiber equals Tᵀ per employment level.
    V_out = randn(2, 2)
    Λ_in  = rand(2, 2)
    V_in  = backward!(stage, V_out, nothing)
    K = parent(stage.kernel)                  # compact (n_out, n_in, dep…) storage
    @test size(K) == (2, 2, 2)                # (to, from, employment)
    @test K[:, :, 1] ≈ permutedims(Tmat(1))
    @test K[:, :, 2] ≈ permutedims(Tmat(2))
    @test K[:, :, 1] != K[:, :, 2]            # genuinely stratum-varying

    # V/Λ duality holds within each employment stratum (r = 0 for Markov), since the
    # kernel acts block-diagonally across :employment.
    Λ_out = forward!(stage, Λ_in)
    for e in 1:2                              # duality per employment column
        @test isapprox(sum(V_in[:, e] .* Λ_in[:, e]),
                       sum(V_out[:, e] .* Λ_out[:, e]); atol = 1e-12)
    end
end

# A constant matrix with no deps must reproduce the pre-refactor numbers exactly.
@testset "MarkovStage — constant matrix matches pre-refactor" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :income, transition_matrix = P)
    V_end = zeros(4, 2); V_end[:, 1] .= 1.0; V_end[:, 2] .= 4.0
    V_start = backward!(stage, V_end, nothing)
    # No dep axes ⇒ the compact storage is a plain (2,2) fiber equal to Pᵀ (seated by backward!).
    @test reshape(parent(stage.kernel), 2, 2) == permutedims(P)
    @test all(isapprox.(V_start[:, 1], 0.7*1.0 + 0.3*4.0; atol = 1e-12))
    @test all(isapprox.(V_start[:, 2], 0.3*1.0 + 0.7*4.0; atol = 1e-12))
end
