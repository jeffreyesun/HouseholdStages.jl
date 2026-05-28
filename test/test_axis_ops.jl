using Test
using HouseholdStages
using HouseholdStages: axis_dim, slices_along, slices_over, fix,
    permute_to_first, permute_to_first!, with_singleton, set_coord

# Toy layout used throughout — (3, 4, 5) array with named axes (:a, :b, :c).
const _LAYOUT = StateLayout(
    StateAxis(:a, continuous_grid([0.0, 1.0, 2.0])),
    StateAxis(:b, discrete_finite([10, 20, 30, 40])),
    StateAxis(:c, categorical([:p, :q, :r, :s, :t])),
)

@testset "axis_ops — axis_dim is pass-through to axis_position" begin
    @test axis_dim(_LAYOUT, :a) == axis_position(_LAYOUT, :a) == 1
    @test axis_dim(_LAYOUT, :b) == axis_position(_LAYOUT, :b) == 2
    @test axis_dim(_LAYOUT, :c) == axis_position(_LAYOUT, :c) == 3
end

@testset "axis_ops — slices_along yields 1-D views along the named axis" begin
    A = reshape(collect(1:60), (3, 4, 5))
    sl = slices_along(A, _LAYOUT, :b)
    # `eachslice(...; dims=2)` over a (3,4,5) array yields 4 slices, each (3,5).
    @test length(sl) == 4
    s_first = first(sl)
    @test size(s_first) == (3, 5)
    @test s_first == A[:, 1, :]

    # Verify against the integer-dim spelling for every named axis.
    for ax in (:a, :b, :c)
        d = axis_dim(_LAYOUT, ax)
        @test collect(slices_along(A, _LAYOUT, ax)) == collect(eachslice(A; dims=d))
    end
end

@testset "axis_ops — slices_over fixes the named axis as free" begin
    A = reshape(collect(1:60), (3, 4, 5))
    sl = slices_over(A, _LAYOUT, :b)
    # Holding :b free, sweeping the other axes (3, 5): 15 slices, each 1-D of length 4.
    @test length(sl) == 15
    s_first = first(sl)
    @test size(s_first) == (4,)
    @test s_first == A[1, :, 1]

    # Round-trip: collecting `slices_over` and reassembling should match.
    for ax in (:a, :b, :c)
        d = axis_dim(_LAYOUT, ax)
        other = Tuple(setdiff(1:3, d))
        @test collect(slices_over(A, _LAYOUT, ax)) == collect(eachslice(A; dims=other))
    end
end

@testset "axis_ops — fix returns a view, not a copy" begin
    A = reshape(collect(1:60), (3, 4, 5))
    v = fix(A, _LAYOUT, :b => 2)
    @test v isa SubArray
    @test size(v) == (3, 5)
    @test v == A[:, 2, :]

    # Mutating the view should propagate back to A.
    v[1, 1] = -999
    @test A[1, 2, 1] == -999
end

@testset "axis_ops — permute_to_first is a no-op when axis is already first" begin
    A = reshape(collect(1:60), (3, 4, 5))
    B = permute_to_first(A, _LAYOUT, :a)
    @test B === A  # same object, no allocation
end

@testset "axis_ops — permute_to_first returns a fresh permuted copy otherwise" begin
    A = reshape(collect(1:60), (3, 4, 5))
    B = permute_to_first(A, _LAYOUT, :b)
    @test !(B === A)
    @test size(B) == (4, 3, 5)
    # The permutation should match permutedims with axis 2 brought to position 1.
    @test B == permutedims(A, (2, 1, 3))

    C = permute_to_first(A, _LAYOUT, :c)
    @test size(C) == (5, 3, 4)
    @test C == permutedims(A, (3, 1, 2))
end

@testset "axis_ops — permute_to_first! writes into dest when permuting" begin
    A = reshape(collect(1:60), (3, 4, 5))
    dest = similar(A, (4, 3, 5))
    B = permute_to_first!(dest, A, _LAYOUT, :b)
    @test B === dest
    @test B == permutedims(A, (2, 1, 3))
end

@testset "axis_ops — permute_to_first! is a no-op (returns A unchanged) when axis is first" begin
    A = reshape(collect(1:60), (3, 4, 5))
    dest = zero(A)  # would-be scratch; should be left untouched
    B = permute_to_first!(dest, A, _LAYOUT, :a)
    @test B === A
    @test all(dest .== 0)  # not written into
end

@testset "axis_ops — with_singleton inserts a 1 at the named axis's position" begin
    # Drop one axis, then re-insert as singleton; shape should round-trip.
    A_flat = reshape(collect(1:12), (3, 4))  # (3, 4)
    sub_layout = StateLayout(
        StateAxis(:a, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:c, categorical([:p, :q, :r, :s])),
    )
    # Insert :a singleton at position 1 → (1, 3, 4)? No: we're using the original
    # full layout :a/:b/:c and "with_singleton" along :b inserts at position 2.
    # Use a layout where :b sits between :a and :c, and the array is (3, 4)
    # representing axes (:a, :c). Inserting singleton at :b's position gives (3, 1, 4).
    A_with = with_singleton(A_flat, _LAYOUT, :b)
    @test size(A_with) == (3, 1, 4)
    @test dropdims(A_with; dims=2) == A_flat

    # And along :a (position 1) and :c (position 3).
    @test size(with_singleton(A_flat, _LAYOUT, :a)) == (1, 3, 4)
    @test size(with_singleton(A_flat, _LAYOUT, :c)) == (3, 4, 1)
end

@testset "axis_ops — set_coord replaces only the named coordinate" begin
    ci = CartesianIndex(1, 2, 3)
    @test set_coord(ci, _LAYOUT, :a => 7) == CartesianIndex(7, 2, 3)
    @test set_coord(ci, _LAYOUT, :b => 7) == CartesianIndex(1, 7, 3)
    @test set_coord(ci, _LAYOUT, :c => 7) == CartesianIndex(1, 2, 7)
end
