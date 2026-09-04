using Test
using HouseholdStages
const H = HouseholdStages

# The payload vocabulary of `STRATIFIED_CONTRACT.md` §4: what a stratified driver hands a fiber op at
# each stratum. Dispatch is the whole validator, so these are tests of dispatch — that each carrier
# resolves to the slice its own conformance says it should, that the degenerate geometries behave,
# and that resolving a slice costs nothing.

@testset "slice protocol — shared payloads pass through" begin
    c = CartesianIndex(1, 2, 3)
    grid = collect(1.0:5)
    @test H._slice(2.5, c, Val(1)) === 2.5
    @test H._slice(nothing, c, Val(1)) === nothing
    @test H._slice(grid, c, Val(2)) === grid          # rank 1 under a rank-3 layout: axis-shared
end

@testset "slice protocol — a per-cell array slices to its fiber" begin
    dims = (4, 3, 2)
    A = reshape(collect(1.0:prod(dims)), dims)
    for adim in 1:3, c in H._strata(dims, Val(adim))
        @test H._slice(A, c, Val(adim)) == A[ntuple(d -> d == adim ? Colon() : c[d], 3)...]
    end
end

@testset "slice protocol — size-1 stratum dims broadcast" begin
    dims = (4, 3, 2)
    B = reshape(collect(1.0:8), (4, 1, 2))            # constant along axis 2
    for c in H._strata(dims, Val(1))
        @test H._slice(B, c, Val(1)) == B[:, 1, c[3]]
    end
end

@testset "slice protocol — the operative extent is the payload's own" begin
    # In and out sides carry different extents on the operative axis; each slices to the length it
    # has, which is what makes a rectangular transition first-class.
    dims = (5, 3)
    V_end   = reshape(collect(1.0:15), (5, 3))
    V_start = reshape(collect(1.0:6),  (2, 3))
    for c in H._strata(dims, Val(1))
        @test length(H._slice(V_end,   c, Val(1))) == 5
        @test length(H._slice(V_start, c, Val(1))) == 2
    end
end

@testset "slice protocol — a MatrixField projects onto its declared deps" begin
    layout = GriddedLayout(:a => GriddedContinuous(collect(1.0:4)),
                           :z => Discrete([1.0, 2.0, 3.0]))
    f = H.matrix_field(Float64, layout, layout, :a, (; z) -> fill(10z, 4, 4))
    H.fill_field!(f, (; z) -> fill(10z, 4, 4), layout, :a, nothing)
    for c in H._strata((4, 3), Val(1))
        @test H._slice(f, c, Val(1)) == fill(10.0 * c[2], 4, 4)
    end
    # A dep-free field is the same fiber at every stratum.
    g = H.matrix_field(Float64, layout, layout, :a, ones(4, 4))
    H.fill_field!(g, ones(4, 4), layout, :a, nothing)
    @test all(H._slice(g, c, Val(1)) == ones(4, 4) for c in H._strata((4, 3), Val(1)))
end

@testset "slice protocol — a ScalarField slices through its bshape" begin
    layout = GriddedLayout(:a => GriddedContinuous(collect(1.0:4)), :z => Discrete([1.0, 2.0]))
    varying = H.ScalarField((; a) -> 2a, layout)      # varies ALONG the operative axis: m = 1
    for c in H._strata((4, 2), Val(1))
        @test collect(H._slice(varying, c, Val(1))) == 2 .* collect(1.0:4)
    end
    @test H._slice(H.ScalarField(3.0, layout), CartesianIndex(1, 1), Val(1)) === 3.0
    # A field constant along the operative axis is `m = 0`; no fiber op consumes one, and which of
    # scalar or one-element fiber the driver should deliver is not yet settled.
    @test_throws AssertionError H._slice(H.ScalarField((; z) -> 5z, layout), CartesianIndex(1, 2), Val(1))
end

@testset "slice protocol — degenerate geometries" begin
    @test H._slice([1.0, 2.0, 3.0], CartesianIndex(1), Val(1)) == [1.0, 2.0, 3.0]   # rank-1 layout
    one_node = reshape([7.0, 8.0], (1, 2))                                          # n_axis == 1
    @test H._slice(one_node, CartesianIndex(1, 2), Val(1)) == [8.0]
    @test isempty(H._slice(zeros(0, 3), CartesianIndex(1, 2), Val(1)))
    @test length(H._strata((4, 3, 2), Val(2))) == 8
end

@testset "slice protocol — entry checks name the shape mistakes" begin
    dims = (4, 3, 2)
    H.check_payloads(dims, 1, zeros(dims), collect(1.0:4), 2.0, nothing)   # all legal
    @test_throws AssertionError H.check_payloads(dims, 1, zeros(4, 3))     # neither shared nor per-cell
    @test_throws AssertionError H.check_payloads(dims, 1, zeros(4, 9, 2))  # stratum extent disagrees
    H.check_payloads(dims, 1, zeros(9, 3, 2))                              # operative extent is free
end

# One monomorphic sweep per payload: a heterogeneous tuple walked inside the timed call would
# allocate on its own account and say nothing about the slices.
sweep(p, dims, adim) = sum(length(H._slice(p, c, adim)) for c in H._strata(dims, adim))

@testset "slice protocol — slicing allocates nothing" begin
    dims   = (4, 3, 2)
    layout = GriddedLayout(:a => GriddedContinuous(collect(1.0:4)),
                           :z => Discrete([1.0, 2.0, 3.0]))
    f = H.matrix_field(Float64, layout, layout, :a, ones(4, 4))
    H.fill_field!(f, ones(4, 4), layout, :a, nothing)

    for p in (reshape(collect(1.0:24), dims),        # per-cell
              reshape(collect(1.0:8), (4, 1, 2)),    # per-cell with a broadcast dim
              collect(1.0:4),                        # axis-shared
              f)                                     # matrix fiber
        sweep(p, dims, Val(1))                       # warm
        @test @allocated(sweep(p, dims, Val(1))) == 0
    end
end
