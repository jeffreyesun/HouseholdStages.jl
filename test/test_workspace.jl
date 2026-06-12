using Test
using HouseholdStages

@testset "allocate — MarkovStage holds its self-describing kernel array directly" begin
    # A modern MarkovStage stores its kernel — the self-describing transition array (a
    # `PermutedDimsArray` over the compact parent) — directly in `.kernel`; the
    # backward/forward output buffers live in `.scratch` (`V_start`/`Λ_end`), no wrapper.
    P = [0.9 0.1; 0.2 0.8]
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:z, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :z, transition_matrix = P)
    @test stage.kernel isa PermutedDimsArray
    @test hasproperty(stage.scratch, :V_start)
    @test hasproperty(stage.scratch, :Λ_end)
end

@testset "allocate — ArgmaxStage holds its single-destination kernel" begin
    layout = GriddedLayout(StateAxis(:s, categorical([:A, :B])))
    stage = ArgmaxStage(layout;
        choice_axis    = :s,
        flow_payoff    = (a; cell, env) -> (a == :B ? 1.0 : 0.0),
        next_state_idx = (cell, a) -> a == :A ? 1 : 2,
    )
    # A modern ArgmaxStage holds its `SingleDestinationKernel` directly in `.kernel`; the
    # integer per-cell destination lives on the kernel (the AD lift reads it there too). The
    # backward/forward output buffers (`V_start`/`Λ_end`) live in `.scratch`.
    @test stage isa HouseholdStages.AbstractModernStage
    @test stage.kernel isa HouseholdStages.SingleDestinationKernel
    @test policy(stage) isa Array{Int}
    @test hasproperty(stage.scratch, :V_start)
    @test hasproperty(stage.scratch, :Λ_end)
    @test size(stage.scratch.V_start) == HouseholdStages.layout_size(layout)
    @test size(stage.scratch.Λ_end)   == HouseholdStages.layout_size(layout)
end

@testset "allocate — LogitChoiceStage operator holds the cost/weight matrices" begin
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoiceStage(layout;
        choice_axis = :a,
        cost_matrix = [0.0 0.5; 0.5 0.0],
        ε           = 0.5,
    )
    # Modern stage: no `Operator`/reward; the kernel IS a `LogitChoiceKernel` carrying the
    # factored pieces. eψC is the dense (n, n) exp(−C/ε) kernel; value_weight/normalizer are
    # full layout-shaped — here just (n,), since the layout is the choice axis only.
    @test stage isa HouseholdStages.AbstractModernStage
    @test stage.kernel isa HouseholdStages.LogitChoiceKernel
    @test stage.kernel.eψC isa AbstractMatrix
    @test size(stage.kernel.eψC) == (2, 2)
    @test size(stage.kernel.value_weight)   == (2,)
    @test size(stage.kernel.normalizer) == (2,)
    @test hasproperty(stage.scratch, :rowmax) && hasproperty(stage.scratch, :kernel_scratch)
end

@testset "allocate — ForgetfulSumStage rides a ones-row marginalising kernel" begin
    layout = GriddedLayout(
        StateAxis(:w, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:t, categorical([:a, :b, :c, :d])),
    )
    stage = ForgetfulSumStage(layout; forget_axis = :t)
    # Modern stage: the kernel IS a ones-row marginalising self-describing array; its
    # compact parent is ones(1, n_forget) (n_out = 1, n_in = n_forget = 4).
    @test stage isa HouseholdStages.AbstractModernStage
    @test stage.kernel isa PermutedDimsArray
    @test (size(parent(stage.kernel), 1), size(parent(stage.kernel), 2)) == (1, 4)
    @test all(stage.kernel .== 1)
    # scratch carries the layout-shaped I/O buffers (V_start full axis, Λ_end resized).
    @test size(stage.scratch.V_start) == (3, 4)
    @test size(stage.scratch.Λ_end)   == (3, 1)
end

@testset "allocate — ChainStage returns per-stage tuple" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = MarkovStage(layout; axis = :z, transition_matrix = P)
    chain = s1 ∘ s2
    @test chain.buffer.stages isa Tuple
    @test length(chain.buffer.stages) == 2
    # `chain.buffer.stages` now holds bundled sub-STAGES. Both are modern
    # MarkovStages: each carries its self-describing kernel array in `.kernel` and its
    # output buffers in `.scratch` (V_start/Λ_end).
    @test chain.buffer.stages[1] isa MarkovStage
    @test chain.buffer.stages[2] isa MarkovStage
    @test chain.buffer.stages[1].kernel isa PermutedDimsArray
    @test chain.buffer.stages[2].kernel isa PermutedDimsArray
    @test hasproperty(chain.buffer.stages[1].scratch, :V_start)
    @test hasproperty(chain.buffer.stages[2].scratch, :V_start)
end

@testset "single-stage backward/forward via buffers" begin
    P = [0.9 0.1; 0.2 0.8]
    layout = GriddedLayout(
        StateAxis(:z, discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = MarkovStage(layout; axis = :z, transition_matrix = P)
    V_end = ones(2, 3)
    V_start = backward!(stage, V_end, nothing)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))

    Λ_start = rand(2, 3); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start)
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
end
