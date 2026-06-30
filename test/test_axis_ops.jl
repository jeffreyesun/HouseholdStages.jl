using Test
using HouseholdStages
using HouseholdStages: fix

# Toy layout used throughout — (3, 4, 5) array with named axes (:a, :b, :c).
const _LAYOUT = GriddedLayout(
    :a => GriddedContinuous([0.0, 1.0, 2.0]),
    :b => Discrete([10, 20, 30, 40]),
    :c => Discrete([:p, :q, :r, :s, :t]),
)

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
