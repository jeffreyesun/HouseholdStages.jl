using Test, HouseholdStages

# The nesting matrix: every way one composite can contain another, each built WITH A LAYOUT-CHANGING
# COMPONENT so the boundaries actually move inside the composite. Square factors hide the whole class
# of layout bugs this exercises, so no case here is square all the way down.
#
# Each case is built on both construction paths — bundled stages, and the spec plus the layouts the
# construction contract says the caller states — and the two must agree on their layouts and on both
# sweeps. The recursive contract's whole content lives on the spec path, so a case tested only from
# bundled stages is half tested.
#
# The construction-error battery at the foot pins one error per assert: the single-axis regrid
# invariant, chain adjacency, product uniformity, matrix-source shape, the two-layout form on a
# non-square chain, and `⊕` twice on one axis. Those errors need a literal matrix to compare shapes
# against; the last testset pins the same under-determined constructions built from `FromEnv` sources
# instead, where nothing is comparable at construction and the refusal lands at the first fill.
#
# Module-wrapped so the matrix's names don't clash with sibling test files' globals.

module NestingMatrixTests

using Test, HouseholdStages

L     = GriddedLayout(:group => Discrete([1]), :x => Discrete([1.0, 2.0, 3.0]))
L_x1  = resize_axis(L, :x, 1)
L2    = grow_axis(L, :group, 2)
L2_x1 = grow_axis(L_x1, :group, 2)
L_g2  = GriddedLayout(:group => Discrete([1, 2]), :x => Discrete([1.0, 2.0, 3.0]))

# The two layout-changing primitives every case is built from: a `3 → 1` collapse and a `1 → 3` grow.
collapse(l = L)  = ForgetfulSumStage(l; axis=:x)
grow(l = L_x1, r = L) = MarkovStage(l, r; axis=:x, transition_matrix=fill(1 / 3, 1, 3))

"""
The interior of a chain whose own components are all primitive: the boundaries they sit between, and
a `nothing` for each of them.
"""
chain_interior(bounds::Tuple) =
    (boundaries = bounds, interiors = ntuple(_ -> nothing, length(bounds) - 1))

"a deterministic layout-shaped probe, distinct at every index so a permuted result differs."
probe(sz::Tuple) = reshape([sin(0.7k) for k in 1:prod(sz)], sz)

"""
Both sweeps on `stage`, each asserted to land on the layout its accessor names. The probes are
non-constant: every stage the matrix is built from maps a constant array to a constant one, so a
constant probe makes the two construction paths agree whatever each of them built.
"""
function round_trip(stage)
    V = backward!(stage, probe(layout_size(end_layout(stage))), NamedTuple())
    Λ = forward!(stage, probe(layout_size(start_layout(stage))))
    @test size(V) == layout_size(start_layout(stage))
    @test size(Λ) == layout_size(end_layout(stage))
    return (copy(V), copy(Λ))
end

"""
One nesting case: the bundled stage and its spec-path twin must report the same two layouts and
produce the same two sweeps.
"""
function nesting_case(bundled, from_spec; start_l, end_l)
    for s in (bundled, from_spec)
        @test start_layout(s) == start_l
        @test end_layout(s)   == end_l
    end
    @test round_trip(bundled) == round_trip(from_spec)
end

@testset "chain ∘ chain" begin
    ch = (IdentityStage(L) ∘ collapse()) ∘ (IdentityStage(L_x1) ∘ grow())
    @test boundaries(ch) == (L, L, L_x1, L_x1, L)
    nesting_case(ch, ChainStage(ch.spec, boundaries(ch)); start_l=L, end_l=L)
end

@testset "chain ⊕ chain" begin
    mkchain() = IdentityStage(L) ∘ collapse()
    p = mkchain() ⊕ mkchain()
    @test interiors(p) == (chain_interior((L, L, L_x1)), chain_interior((L, L, L_x1)))
    nesting_case(p, ProductStage(p.spec, L2, L2_x1, interiors(p)); start_l=L2, end_l=L2_x1)
end

@testset "product ∘ product" begin
    ch = (collapse() ⊕ collapse()) ∘ (grow() ⊕ grow())
    @test boundaries(ch) == (L2, L2_x1, L2)
    nesting_case(ch, ChainStage(ch.spec, boundaries(ch)); start_l=L2, end_l=L2)
end

@testset "product ∘ primitive" begin
    ch = (collapse() ⊕ collapse()) ∘ grow(L2_x1, L2)
    @test boundaries(ch) == (L2, L2_x1, L2)
    nesting_case(ch, ChainStage(ch.spec, boundaries(ch)); start_l=L2, end_l=L2)
end

@testset "chain containing ⊕" begin
    ch = IdentityStage(L2) ∘ (collapse() ⊕ collapse()) ∘ IdentityStage(L2_x1)
    @test boundaries(ch) == (L2, L2, L2_x1, L2_x1)
    nesting_case(ch, ChainStage(ch.spec, boundaries(ch)); start_l=L2, end_l=L2_x1)
end

@testset "chain ⊕ primitive" begin
    p = (IdentityStage(L) ∘ collapse()) ⊕ collapse()
    @test interiors(p) == (chain_interior((L, L, L_x1)), nothing)
    nesting_case(p, ProductStage(p.spec, L2, L2_x1, interiors(p)); start_l=L2, end_l=L2_x1)
end

@testset "heterogeneous ⊕ — Markov ⊕ Identity, and a regridding pair of different specs" begin
    P = fill(1 / 3, 3, 3)
    p = MarkovStage(L; axis=:x, transition_matrix=P) ⊕ IdentityStage(L)
    nesting_case(p, ProductStage(p.spec, L2, L2); start_l=L2, end_l=L2)

    # The same heterogeneity with both factors regridding: a Markov collapse beside a ForgetfulSum.
    q = MarkovStage(L, L_x1; axis=:x, transition_matrix=ones(3, 1)) ⊕ collapse()
    nesting_case(q, ProductStage(q.spec, L2, L2_x1); start_l=L2, end_l=L2_x1)
end

@testset "product over a chain whose interior regrids — the `replicate_age` shape" begin
    # The factor chain is size-1 on `:x` at BOTH ends and visits size 3 inside, so its interior is
    # exactly what the product cannot determine from its own two layouts. This is the shape the two
    # de Nardi models build.
    inner() = ChainStage((grow(), collapse()))
    @test boundaries(inner()) == (L_x1, L, L_x1)

    p = replicate_age(inner(), 3; axis=:group)
    start_l = grow_axis(L_x1, :group, 3)
    @test interiors(p) == ntuple(_ -> chain_interior((L_x1, L, L_x1)), 3)
    nesting_case(p, ProductStage(p.spec, start_l, start_l, interiors(p)); start_l, end_l=start_l)

    # `replicate_age` is handed ONE object N times; the factors must be N independent rebuilds.
    comps = p.buffer.components
    @test length(comps) == 3
    @test comps[1] !== comps[2] !== comps[3]
    @test all(c -> boundaries(c) == (L_x1, L, L_x1), comps)
end

@testset "chain containing a product over regridding chains — the de Nardi shape" begin
    # Two levels of composite below the chain: the chain states the product's interior, and the
    # product's interior states each factor chain's joins. Nothing in the pass-down reaches those
    # joins, so a contract implemented one level deep cannot rebuild this from the spec.
    factor() = ChainStage((grow(), collapse()))              # `:x` runs 1 → 3 → 1 between size-1 ends
    G        = grow_axis(L_x1, :group, 3)
    ch       = IdentityStage(G) ∘ replicate_age(factor(), 3; axis=:group)

    @test boundaries(ch) == (G, G, G)
    @test interiors(ch) == (nothing, ntuple(_ -> chain_interior((L_x1, L, L_x1)), 3))
    nesting_case(ch, ChainStage(ch.spec, boundaries(ch), interiors(ch)); start_l=G, end_l=G)

    # The transition path rebuilds one chain per period from the spec, so it needs the whole of that
    # construction data. Every leaf here is linear, so its kernel does not move with `V` and the path
    # is the plain iterated sweep.
    V_T, Λ_0 = probe(layout_size(end_layout(ch))), probe(layout_size(start_layout(ch)))
    env_path = [NamedTuple() for _ in 1:3]
    (;V_path, Λ_path) = solve_transition_given_env_path!(ch, env_path; Λ_0, V_T)

    V, Λ = copy(V_T), copy(Λ_0)
    for _ in env_path; V = copy(backward!(ch, V, NamedTuple())); end
    for _ in env_path; Λ = copy(forward!(ch, Λ));                 end
    @test V_path[1]   ≈ V
    @test Λ_path[end] ≈ Λ
    @test sum(Λ_path[end]) ≈ sum(Λ_0)                        # every leaf here conserves mass

    # Leaving the product's interior unstated is under-determined. The factor chain's two ends agree,
    # so the refusal lands on the leaf whose two layouts came out equal rather than at the chain.
    err = @test_throws AssertionError ChainStage(ch.spec, boundaries(ch))
    @test occursin("inside a composite", err.value.msg)
end

@testset "construction errors — one per assert, each naming the fix" begin
    # The single-axis regrid invariant: a primitive may move only its operative axis.
    err = @test_throws AssertionError MarkovStage(L, L_g2; axis=:x, transition_matrix=fill(1 / 3, 3, 3))
    @test occursin("operative axis", err.value.msg)
    err = @test_throws AssertionError IdentityStage(L, L_x1)
    @test occursin("no operative axis", err.value.msg)

    # …asserted from the INNER constructor, so assembling the six fields positionally — the form a
    # relocation rebuilds through — cannot slip a mismatched pair past it, with or without the
    # type parameters.
    s = IdentityStage(L)
    err = @test_throws AssertionError typeof(s).name.wrapper(s.spec, L, L_x1, s.kernel, s.scratch, s.cache)
    @test occursin("no operative axis", err.value.msg)
    err = @test_throws AssertionError typeof(s)(s.spec, L, L_x1, s.kernel, s.scratch, s.cache)
    @test occursin("no operative axis", err.value.msg)
    # The pair it was built from still goes through, unchanged.
    @test start_layout(typeof(s).name.wrapper(s.spec, L, L, s.kernel, s.scratch, s.cache)) == L

    # A literal matrix source of the wrong shape is refused at construction, by the fill that seats
    # it. The shapes are quoted as STORED: `MarkovStage` keeps `K = Tᵀ`, on a `(n_end, n_start)` face.
    err = @test_throws AssertionError MarkovStage(L, L_x1; axis=:x, transition_matrix=ones(3, 3))
    @test occursin("fill_field!: source fiber (3, 3) ≠ field face (1, 3)", err.value.msg)

    # Chain adjacency, at both entry points.
    err = @test_throws AssertionError ChainStage((collapse(), IdentityStage(L)))
    @test occursin("successor", err.value.msg)
    @test_throws AssertionError collapse() ∘ IdentityStage(L)
    err = @test_throws AssertionError ChainStage((IdentityStage(L) ∘ IdentityStage(L)).spec, (L, L))
    @test occursin("boundary layouts", err.value.msg)

    # Product uniformity: the factors sit between the same two boundaries.
    err = @test_throws AssertionError product(IdentityStage(L), IdentityStage(L_x1); axis=:group)
    @test occursin("in parallel between the same two boundaries", err.value.msg)

    # The product-level layouts carry the product axis at `n`, refused too small at either end …
    two   = (IdentityStage(L) ⊕ IdentityStage(L)).spec
    L3    = grow_axis(L, :group, 3)
    err   = @test_throws AssertionError ProductStage(two, L, L)
    @test occursin("one level per component", err.value.msg)
    err = @test_throws AssertionError ProductStage(two, L2, L)
    @test occursin("`group` axis is size 1", err.value.msg)
    # … and too large, which would otherwise construct and leave the surplus slices at zero.
    @test_throws AssertionError ProductStage(two, L3, L3)
    err = @test_throws AssertionError ProductStage(two, L2, L3)
    @test occursin("`group` axis is size 3", err.value.msg)

    # A regridding matrix source handed a single layout: the message names the two-layout form.
    err = @test_throws AssertionError ArgmaxStage(L; axis=:x, reward=zeros(3, 1))
    @test occursin("start_layout, end_layout", err.value.msg)

    # The two-layout form on a chain that regrids, reached directly and through a product component
    # whose interior was left unstated.
    ch = IdentityStage(L) ∘ collapse()
    err = @test_throws AssertionError bundle(ch.spec, L, L_x1)
    @test occursin("boundary sequence", err.value.msg)
    err = @test_throws AssertionError ProductStage((ch ⊕ ch).spec, L2, L2_x1)
    @test occursin("boundary sequence", err.value.msg)

    # `⊕` twice on the same axis stays refused, naming the fixed-layout invariant.
    p = IdentityStage(L) ⊕ IdentityStage(L)
    err = @test_throws AssertionError p ⊕ p
    @test occursin("fixed-layout invariant", err.value.msg)
end

@testset "an under-determined interior built from `FromEnv` sources — square, then refused at fill" begin
    # A literal matrix commits its stage to a fiber shape, so leaving the interior unstated puts two
    # known shapes side by side and construction quotes them both. A `FromEnv` source commits to
    # nothing: the same two constructions come out square all the way down, and the first fill is
    # where the fiber `env` actually holds meets the face the square layouts allocated.
    grow_env()     = MarkovStage(L_x1, L;    axis=:x, transition_matrix=FromEnv(:T))
    collapse_env() = MarkovStage(L,    L_x1; axis=:x, transition_matrix=FromEnv(:S))
    env            = (; T = fill(1 / 3, 1, 3), S = ones(3, 1))

    # The refusal a stage that came out square raises the first time it is swept.
    fill_refusal(stage, l) = @test_throws AssertionError backward!(stage, ones(layout_size(l)), env)

    # `:x` runs 1 → 3 → 1 between size-1 ends, which the stated boundary sequence builds and runs.
    ch = grow_env() ∘ collapse_env()
    @test boundaries(ch) == (L_x1, L, L_x1)
    @test size(backward!(ch, ones(layout_size(L_x1)), env)) == layout_size(L_x1)

    # The two-layout form is legal — the ends agree — and states an interior that does not move.
    square = bundle(ch.spec, L_x1, L_x1)
    @test boundaries(square) == (L_x1, L_x1, L_x1)
    err = fill_refusal(square, L_x1)
    @test occursin("fill_field!: source fiber", err.value.msg)
    @test occursin("field face (1, 1)", err.value.msg)

    # The same through a product whose interior is left unstated — the de Nardi shape, one level
    # deeper, where a literal source raises at the leaf whose two layouts came out equal.
    G   = grow_axis(L_x1, :group, 3)
    ch2 = IdentityStage(G) ∘ replicate_age(ChainStage((grow_env(), collapse_env())), 3; axis=:group)
    @test boundaries(ch2) == (G, G, G)

    square2 = ChainStage(ch2.spec, boundaries(ch2))
    @test boundaries(square2) == (G, G, G)
    @test all(c -> boundaries(c) == (L_x1, L_x1, L_x1), square2.buffer.stages[2].buffer.components)
    fill_refusal(square2, G)
end

end # module
