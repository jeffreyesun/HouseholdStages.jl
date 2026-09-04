using Test
using HouseholdStages

# ProductStage operates along a PRE-EXISTING singleton axis (resized 1 → n), not an appended one —
# the fixed-layout invariant. Every factor's layout carries the product axis at size 1; the product
# resizes it to the number of factors, one factor per slice (at any axis position).

@testset "product — uniform 2-component, axis = :group" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = product(s, s; axis = :group)
    @test ps isa ProductStage
    @test ps.spec.axis === :group
    @test length(ps.spec.components) == 2
    @test layout_size(start_layout(ps)) == (2, 2)
    @test layout_size(end_layout(ps)) == (2, 2)
end

@testset "product — backward & forward run per-component" begin
    P = [0.8 0.2; 0.2 0.8]
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = product(s, s; axis = :group)

    V_end = ones(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))

    Λ_start = rand(2, 2)
    Λ_start[:, 1] ./= sum(Λ_start[:, 1]); Λ_start[:, 2] ./= sum(Λ_start[:, 2])
    Λ_end = forward!(ps, Λ_start)
    @test isapprox(sum(Λ_end[:, 1]), 1.0; atol = 1e-12)
    @test isapprox(sum(Λ_end[:, 2]), 1.0; atol = 1e-12)
end

@testset "⊕ infix operator" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    P = [0.5 0.5; 0.5 0.5]
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = s1 ⊕ s2
    @test ps isa ProductStage
    @test length(ps.spec.components) == 2
end

@testset "replicate_age — N copies along :age" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :age => Discrete([1]))
    P = [0.5 0.5; 0.5 0.5]
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    ages = replicate_age(s, 5)
    @test ages.spec.axis === :age
    @test length(ages.spec.components) == 5
    @test layout_size(start_layout(ages)) == (2, 5)
end

# Each product-axis slice equals the component run standalone (on a size-1 slice, with i:i to keep
# the axis), with duality and per-component mass conservation.
@testset "product — fused slices match standalone components (uniform MarkovStage)" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    P = [0.7 0.3; 0.3 0.7]
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = product(s1, s2; axis = :group)

    V_end = randn(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test V_start[:, 1:1] ≈ backward!(s1, V_end[:, 1:1], nothing)
    @test V_start[:, 2:2] ≈ backward!(s2, V_end[:, 2:2], nothing)

    Λ_start = rand(2, 2)
    Λ_start[:, 1] ./= sum(Λ_start[:, 1]); Λ_start[:, 2] ./= sum(Λ_start[:, 2])
    Λ_end = forward!(ps, Λ_start)
    @test Λ_end[:, 1:1] ≈ forward!(s1, Λ_start[:, 1:1])
    @test Λ_end[:, 2:2] ≈ forward!(s2, Λ_start[:, 2:2])
    @test isapprox(sum(V_start .* Λ_start), sum(V_end .* Λ_end); atol = 1e-12)
    @test isapprox(sum(Λ_end[:, 1]), 1.0; atol = 1e-12)
    @test isapprox(sum(Λ_end[:, 2]), 1.0; atol = 1e-12)
end

@testset "product — along a NON-final axis (group first)" begin
    # The product axis need not be last: here :group is the FIRST layout axis.
    layout = GriddedLayout(:group => Discrete([1]), :z => Discrete([0.5, 1.5]))
    P = [0.7 0.3; 0.3 0.7]
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = product(s1, s2; axis = :group)
    @test layout_size(start_layout(ps)) == (2, 2)        # (group, z)

    V_end = randn(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test V_start[1:1, :] ≈ backward!(s1, V_end[1:1, :], nothing)
    @test V_start[2:2, :] ≈ backward!(s2, V_end[2:2, :], nothing)
    Λ_start = rand(2, 2)
    Λ_end = forward!(ps, Λ_start)
    @test sum(Λ_end) ≈ sum(Λ_start) atol = 1e-12
end

@testset "product — IdentityStage components pass through per slice" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    s1 = IdentityStage(layout); s2 = IdentityStage(layout)
    ps = product(s1, s2; axis = :group)

    V_end = randn(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test V_start ≈ V_end
    Λ_start = rand(2, 2)
    Λ_end = forward!(ps, Λ_start)
    @test Λ_end ≈ Λ_start
    @test isapprox(sum(V_start .* Λ_start), sum(V_end .* Λ_end); atol = 1e-12)
end

@testset "product — ForgetfulSumStage components fuse correctly" begin
    layout = GriddedLayout(
        :w => GriddedContinuous([0.0, 1.0, 2.0]),
        :t => Discrete([:a, :b]),
        :group => Discrete([1]),
    )
    s1 = ForgetfulSumStage(layout; axis = :t)
    s2 = ForgetfulSumStage(layout; axis = :t)
    ps = product(s1, s2; axis = :group)
    pdim = 3                                            # :group axis position

    V_end = randn(layout_size(end_layout(ps)))      # (w, t = 1, group = 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test selectdim(V_start, pdim, 1:1) ≈ backward!(s1, selectdim(V_end, pdim, 1:1), nothing)
    @test selectdim(V_start, pdim, 2:2) ≈ backward!(s2, selectdim(V_end, pdim, 2:2), nothing)

    Λ_start = rand(layout_size(start_layout(ps))...)   # (w, t, group)
    Λ_end = forward!(ps, Λ_start)
    @test selectdim(Λ_end, pdim, 1:1) ≈ forward!(s1, copy(selectdim(Λ_start, pdim, 1:1)))
    @test selectdim(Λ_end, pdim, 2:2) ≈ forward!(s2, copy(selectdim(Λ_start, pdim, 2:2)))
    @test isapprox(sum(Λ_end), sum(Λ_start); atol = 1e-12)
end

@testset "product — choice (Argmax) stages fuse correctly" begin
    layout = GriddedLayout(:a => Discrete([1, 2]), :group => Discrete([1]))
    reward = [1.0 1.0; 2.0 2.0]                         # M[after, before] = value[after]
    s1 = ArgmaxStage(layout; axis = :a, reward = reward)
    s2 = ArgmaxStage(layout; axis = :a, reward = reward)
    ps = product(s1, s2; axis = :group)

    V_end = randn(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test V_start[:, 1:1] ≈ backward!(s1, V_end[:, 1:1], NamedTuple())
    @test V_start[:, 2:2] ≈ backward!(s2, V_end[:, 2:2], NamedTuple())
    @test policy(ps.buffer.components[1]) isa Array

    Λ_start = rand(2, 2)
    Λ_end = forward!(ps, Λ_start)
    @test Λ_end[:, 1:1] ≈ forward!(s1, Λ_start[:, 1:1])
    @test Λ_end[:, 2:2] ≈ forward!(s2, Λ_start[:, 2:2])
end

@testset "product — heterogeneous factors run side by side" begin
    # Factors need only agree on their two layouts; their specs are free to differ.
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    P = [0.5 0.5; 0.5 0.5]
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = IdentityStage(layout)
    ps = product(s1, s2; axis = :group)

    V_end = randn(2, 2)
    V_start = backward!(ps, V_end, NamedTuple())
    @test V_start[:, 1:1] ≈ backward!(s1, V_end[:, 1:1], NamedTuple())
    @test V_start[:, 2:2] ≈ V_end[:, 2:2]
end

@testset "product — missing product axis errors (no introduce)" begin
    # The product axis must exist at size 1 in the factors; a missing one is rejected (no introduce).
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))   # no :group axis
    P = [0.5 0.5; 0.5 0.5]
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    @test_throws AssertionError product(s, s; axis = :group)
end

@testset "product — env slice unions component slices" begin
    layout = GriddedLayout(:a => Discrete([1, 2]), :age => Discrete([1]))
    C  = [0.0 0.5; 0.5 0.0]
    s1 = LogitChoiceStage(layout; axis = :a, cost_matrix = C, ε = FromEnv(:ξ1))
    s2 = LogitChoiceStage(layout; axis = :a, cost_matrix = C, ε = FromEnv(:ξ2))
    ps = product(s1, s2; axis = :age)
    slice = effective_env_slice(ps)
    @test :ξ1 in slice
    @test :ξ2 in slice
end
