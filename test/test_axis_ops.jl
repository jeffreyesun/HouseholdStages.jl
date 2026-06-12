using Test
using HouseholdStages
using HouseholdStages: axis_dim, fix, set_coord

# Toy layout used throughout — (3, 4, 5) array with named axes (:a, :b, :c).
const _LAYOUT = GriddedLayout(
    StateAxis(:a, continuous_grid([0.0, 1.0, 2.0])),
    StateAxis(:b, discrete_finite([10, 20, 30, 40])),
    StateAxis(:c, categorical([:p, :q, :r, :s, :t])),
)

@testset "axis_ops — axis_dim is pass-through to axis_position" begin
    @test axis_dim(_LAYOUT, :a) == axis_position(_LAYOUT, :a) == 1
    @test axis_dim(_LAYOUT, :b) == axis_position(_LAYOUT, :b) == 2
    @test axis_dim(_LAYOUT, :c) == axis_position(_LAYOUT, :c) == 3
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

@testset "axis_ops — set_coord replaces only the named coordinate" begin
    ci = CartesianIndex(1, 2, 3)
    @test set_coord(ci, _LAYOUT, :a => 7) == CartesianIndex(7, 2, 3)
    @test set_coord(ci, _LAYOUT, :b => 7) == CartesianIndex(1, 7, 3)
    @test set_coord(ci, _LAYOUT, :c => 7) == CartesianIndex(1, 2, 7)
end
