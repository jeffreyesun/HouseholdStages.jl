using Test, HouseholdStages

# The public accessor set is UNIFORM across the two supertypes: whether a stage is primitive or
# composite is an implementation detail, and nothing a caller can reach distinguishes them. One
# battery — `start_layout`, `end_layout`, `layout`, `V_start_buffer`, `Λ_end_buffer`,
# `effective_env_slice`, `.spec`, `backward!`, `forward!` — therefore runs over a primitive, a
# `ChainStage` and a `ProductStage`, in a square and a regridding version of each, and asserts the
# same answers from all six. The `layout` property is exercised in both directions along the way: it
# returns for the square stages and throws for the regridding ones.
#
# Module-wrapped so the battery's names don't clash with sibling test files' globals — and so that
# `layout` here is the exported accessor rather than some earlier file's leftover binding.

module StageAccessorTests

using Test, HouseholdStages

L     = GriddedLayout(:group => Discrete([1]), :x => Discrete([1.0, 2.0, 3.0]))
L_x1  = resize_axis(L, :x, 1)
L2    = grow_axis(L, :group, 2)
L2_x1 = grow_axis(L_x1, :group, 2)

# A square and a `3 → 1` collapsing MarkovStage, both reading their transition from `env` so the
# battery's env slice has content on every stage kind.
square_markov(key = :P)   = MarkovStage(L; axis=:x, transition_matrix=FromEnv(key))
collapse_markov(key = :Q) = MarkovStage(L, L_x1; axis=:x, transition_matrix=FromEnv(key))
accessor_env = (P = fill(1 / 3, 3, 3), Q = ones(3, 1))

"""
The full public accessor battery on one stage: the two layouts, the asserting `layout` property, the
two endpoint buffers, the env slice, the spec, and a `backward!`/`forward!` round trip that must land
in those same endpoint buffers.
"""
function accessor_battery(stage; start_l, end_l, env_slice)
    @test start_layout(stage) == start_l
    @test end_layout(stage)   == end_l

    # A2.3 — the singular property, legal only for a square stage.
    if start_l == end_l
        @test layout(stage) == start_l
    else
        @test_throws AssertionError layout(stage)
    end

    # A4.7 — the layouts describe the buffers.
    @test size(V_start_buffer(stage)) == layout_size(start_l)
    @test size(Λ_end_buffer(stage))   == layout_size(end_l)

    @test effective_env_slice(stage) == env_slice
    @test effective_env_slice(stage) == effective_env_slice(stage.spec)
    @test stage.spec isa AbstractStageSpec

    V_start = backward!(stage, zeros(layout_size(end_l)), accessor_env)
    @test size(V_start) == layout_size(start_l)
    @test V_start === V_start_buffer(stage)          # the sweep writes into the endpoint buffer

    Λ_end = forward!(stage, ones(layout_size(start_l)))
    @test size(Λ_end) == layout_size(end_l)
    @test Λ_end === Λ_end_buffer(stage)
end

@testset "Accessor uniformity — square primitive / chain / product" begin
    accessor_battery(square_markov(); start_l=L, end_l=L, env_slice=(:P,))
    accessor_battery(IdentityStage(L) ∘ square_markov(); start_l=L, end_l=L, env_slice=(:P,))
    accessor_battery(square_markov() ⊕ square_markov(); start_l=L2, end_l=L2, env_slice=(:P,))
end

@testset "Accessor uniformity — regridding primitive / chain / product" begin
    accessor_battery(collapse_markov(); start_l=L, end_l=L_x1, env_slice=(:Q,))
    accessor_battery(IdentityStage(L) ∘ collapse_markov(); start_l=L, end_l=L_x1, env_slice=(:Q,))
    accessor_battery(collapse_markov() ⊕ collapse_markov(); start_l=L2, end_l=L2_x1, env_slice=(:Q,))
end

@testset "Accessor uniformity — a composite's env slice is the union over its components" begin
    @test Set(effective_env_slice(square_markov(:P) ∘ square_markov(:P2))) == Set((:P, :P2))
    @test Set(effective_env_slice(square_markov(:P) ⊕ square_markov(:P2))) == Set((:P, :P2))
end

# A4.7 over the whole primitive set. `W` carries a continuous axis for the interpolating and
# continuous-choice stages. Each stage appears in every direction its own restriction admits: Markov
# both ways, argmax and logit growing the choice axis, and the square-coded ones (Bucket B item B5)
# square.
W = GriddedLayout(:group => Discrete([1]), :w => GriddedContinuous([0.0, 1.0, 2.0, 3.0, 4.0]))
_mix_K(p) = [1 - p p / 2 p / 2; p / 2 1 - p p / 2; p / 2 p / 2 1 - p]

"The `@definestage` stage type names, read off the trait the macro emits — one method each."
definestage_names() =
    Set(nameof(m.sig.parameters[2].var.ub) for m in methods(HouseholdStages.spec_type))

@testset "A4.7 — layouts match buffers, over every `@definestage` primitive" begin
    primitives = (IdentityStage(L),
                  UtilityStage(L; utility=1.0),
                  PointwiseScaleStage(L; backward=0.9, forward=1.0),
                  TimeDiscountingStage(L; β=0.95),
                  EntryStage(L; entry=zeros(layout_size(L))),
                  MarkovStage(L; axis=:x, transition_matrix=fill(1 / 3, 3, 3)),
                  MarkovStage(L, L_x1; axis=:x, transition_matrix=ones(3, 1)),
                  MarkovStage(L_x1, L; axis=:x, transition_matrix=fill(1 / 3, 1, 3)),
                  ArgmaxStage(L; axis=:x, reward=zeros(3, 3)),
                  ArgmaxStage(L_x1, L; axis=:x, reward=zeros(3, 1)),
                  LogitChoiceStage(L; axis=:x, cost_matrix=zeros(3, 3), ε=0.5),
                  LogitChoiceStage(L_x1, L; axis=:x, cost_matrix=zeros(1, 3), ε=0.5),
                  MixingStage(L; axis=:x, K_A=_mix_K(0.2), K_B=_mix_K(0.8), cost_curvature=2.0),
                  DiscreteMoveStage(W; axis=:w, destination=(; w) -> 0.5w + 1),
                  DeterministicContinuousStage(W; axis=:w, destination=(; w) -> 0.5w + 1),
                  ContinuousArgmaxStage(W; axis=:w, reward=(w, w_next) -> -0.5 * (w_next - 0.5w)^2),
                  MeanPreservingSpreadStage(W; axis=:w, θ_max=1.0, cost=(θ; env) -> θ^2),
                  GaussianLoadingStage(W; axis=:w, anchor=1.02, increment_mean=0.03,
                                       increment_sd=0.2, cost=(θ; env) -> 0.01θ^2))
    composites = (ForgetfulSumStage(L; axis=:x),
                  IdentityStage(L) ∘ ForgetfulSumStage(L; axis=:x),
                  ForgetfulSumStage(L; axis=:x) ⊕ ForgetfulSumStage(L; axis=:x))
    for s in (primitives..., composites...)
        @test size(V_start_buffer(s)) == layout_size(start_layout(s))
        @test size(Λ_end_buffer(s))   == layout_size(end_layout(s))
    end

    # The battery is the whole primitive set, not a sample of it: a fourteenth `@definestage` fails
    # here until it is added above.
    @test all(s -> s isa HouseholdStages.AbstractPrimitiveStage, primitives)
    @test Set(nameof(typeof(s).name.wrapper) for s in primitives) == definestage_names()
end

end # module
