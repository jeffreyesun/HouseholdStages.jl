using Test
using HouseholdStages

@testset "product — uniform 2-component, axis = :group" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = GriddedLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = product(s, s; axis = :group)
    @test ps isa ProductStage
    @test ps.spec.axis === :group
    @test length(ps.spec.components) == 2
    @test layout_size(input_layout(ps)) == (2, 2)
    @test layout_size(output_layout(ps)) == (2, 2)
end

@testset "product — backward & forward run per-component" begin
    P = [0.8 0.2; 0.2 0.8]
    layout = GriddedLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = product(s, s; axis = :group)

    V_end = ones(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))

    # Per-component Λ_start summing to 1 each → Λ_end sums per component.
    Λ_start = rand(2, 2)
    Λ_start[:, 1] ./= sum(Λ_start[:, 1])
    Λ_start[:, 2] ./= sum(Λ_start[:, 2])
    Λ_end = forward!(ps, Λ_start)
    @test isapprox(sum(Λ_end[:, 1]), 1.0; atol = 1e-12)
    @test isapprox(sum(Λ_end[:, 2]), 1.0; atol = 1e-12)
end

@testset "× infix operator" begin
    layout = GriddedLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.5 0.5; 0.5 0.5]
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = s1 × s2
    @test ps isa ProductStage
    @test length(ps.spec.components) == 2
end

@testset "replicate_age — N copies along :age" begin
    layout = GriddedLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.5 0.5; 0.5 0.5]
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    ages = replicate_age(s, 5)
    @test ages.spec.axis === :age
    @test length(ages.spec.components) == 5
    @test layout_size(input_layout(ages)) == (2, 5)
end

# The product no longer view-stitches component buffers into the fused tensor;
# it copies each component's result into the matching product-axis slice. These
# testsets assert that behavior numerically: the fused slice for component `i`
# equals running that component standalone, with duality and mass conservation.
@testset "product — fused slices match standalone components (uniform MarkovStage)" begin
    layout = GriddedLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.7 0.3; 0.3 0.7]
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = product(s1, s2; axis = :group)

    V_end = randn(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    # Each product-axis slice equals the component run standalone on that slice.
    @test V_start[:, 1] ≈ backward!(s1, V_end[:, 1], nothing)
    @test V_start[:, 2] ≈ backward!(s2, V_end[:, 2], nothing)

    Λ_start = rand(2, 2)
    Λ_start[:, 1] ./= sum(Λ_start[:, 1]); Λ_start[:, 2] ./= sum(Λ_start[:, 2])
    Λ_end = forward!(ps, Λ_start)
    @test Λ_end[:, 1] ≈ forward!(s1, Λ_start[:, 1])
    @test Λ_end[:, 2] ≈ forward!(s2, Λ_start[:, 2])
    # Duality on the fused pair (r = 0 for pure Markov) and per-component mass.
    @test isapprox(sum(V_start .* Λ_start), sum(V_end .* Λ_end); atol = 1e-12)
    @test isapprox(sum(Λ_end[:, 1]), 1.0; atol = 1e-12)
    @test isapprox(sum(Λ_end[:, 2]), 1.0; atol = 1e-12)
end

@testset "product — IdentityStage components pass through per slice" begin
    layout = GriddedLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s1 = IdentityStage(layout)
    s2 = IdentityStage(layout)
    ps = product(s1, s2; axis = :group)

    V_end = randn(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test V_start ≈ V_end                              # identity is a pass-through

    Λ_start = rand(2, 2)
    Λ_end = forward!(ps, Λ_start)
    @test Λ_end ≈ Λ_start
    @test isapprox(sum(V_start .* Λ_start), sum(V_end .* Λ_end); atol = 1e-12)
end

@testset "product — ForgetfulSumStage components fuse correctly" begin
    layout = GriddedLayout(
        StateAxis(:w, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:t, categorical([:a, :b])),
    )
    s1 = ForgetfulSumStage(layout; forget_axis = :t)
    s2 = ForgetfulSumStage(layout; forget_axis = :t)
    ps = product(s1, s2; axis = :group)

    # V_end is on the OUTPUT layout (forget axis collapsed to a single level);
    # backward! broadcasts it back. Match each component standalone on its slice.
    V_end = randn(layout_size(output_layout(ps)))      # (w, t=1, group)
    V_start = backward!(ps, V_end, NamedTuple())
    Nin = ndims(V_start)
    @test selectdim(V_start, Nin, 1) ≈ backward!(s1, selectdim(V_end, Nin, 1), nothing)
    @test selectdim(V_start, Nin, 2) ≈ backward!(s2, selectdim(V_end, Nin, 2), nothing)

    Λ_start = rand(layout_size(input_layout(ps))...)   # (w, t, group)
    Λ_end = forward!(ps, Λ_start)
    Nls = ndims(Λ_start)
    @test selectdim(Λ_end, ndims(Λ_end), 1) ≈ forward!(s1, copy(selectdim(Λ_start, Nls, 1)))
    @test selectdim(Λ_end, ndims(Λ_end), 2) ≈ forward!(s2, copy(selectdim(Λ_start, Nls, 2)))
    # Mass is conserved by the forgetful sum (it only marginalises).
    @test isapprox(sum(Λ_end), sum(Λ_start); atol = 1e-12)
end

@testset "product — choice (Argmax) stages fuse correctly" begin
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2])))
    fp  = (cell, a; env) -> Float64(a)
    nsx = (cell, a) -> a
    s1 = ArgmaxStage(layout; choice_axis = :a, flow_payoff = fp, next_state_idx = nsx)
    s2 = ArgmaxStage(layout; choice_axis = :a, flow_payoff = fp, next_state_idx = nsx)
    ps = product(s1, s2; axis = :group)

    V_end = randn(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test V_start[:, 1] ≈ backward!(s1, V_end[:, 1], NamedTuple())
    @test V_start[:, 2] ≈ backward!(s2, V_end[:, 2], NamedTuple())
    # The product stays legacy (`ps.buffer.components`), but each component is a
    # modern ArgmaxStage carrying its policy on the kernel's selection fiber.
    @test policy(ps.buffer.components[1]) isa Array

    Λ_start = rand(2, 2)
    Λ_end = forward!(ps, Λ_start)
    @test Λ_end[:, 1] ≈ forward!(s1, Λ_start[:, 1])
    @test Λ_end[:, 2] ≈ forward!(s2, Λ_start[:, 2])
end

@testset "product — heterogeneous types error" begin
    layout = GriddedLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.5 0.5; 0.5 0.5]
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = IdentityStage(layout)
    @test_throws AssertionError product(s1, s2; axis = :group)
end

@testset "product — env slice unions component slices" begin
    # Two LogitChoiceStages with the same cost matrix but distinct
    # env-resolved ε keys: the product's env slice should union them.
    layout = GriddedLayout(StateAxis(:a, discrete_finite([1, 2])))
    C  = [0.0 0.5; 0.5 0.0]
    s1 = LogitChoiceStage(layout;
        choice_axis = :a, cost_matrix = C, ε = FromEnv(:ξ1),
    )
    s2 = LogitChoiceStage(layout;
        choice_axis = :a, cost_matrix = C, ε = FromEnv(:ξ2),
    )
    ps = product(s1, s2; axis = :age)
    slice = effective_env_slice(ps)
    @test :ξ1 in slice
    @test :ξ2 in slice
end
