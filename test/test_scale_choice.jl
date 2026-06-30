using Test
using HouseholdStages
using Statistics

# ScaleChoiceStage (rung c): choose the dispersion θ of a mean-preserving spread. Concave V ⇒
# insure (θ↓), convex V ⇒ gamble (θ↑) — driven purely by the curvature of V (zero cost here).

xs    = collect(0.0:1.0:10.0)          # evenly spaced so interior MPS rows preserve the mean exactly
block = GriddedLayout(:x => GriddedContinuous(xs), :scale => Discrete([1]))
shocks, weights = [-1.0, 1.0], [0.5, 0.5]
dispersions     = [0.0, 1.0, 2.0]      # θ=0 is the no-spread (identity) corner

@testset "scale — MPS kernels preserve interior mean, grow variance" begin
    grid = xs
    Ks   = [HouseholdStages._mps_kernel(grid, θ, shocks, weights) for θ in dispersions]
    for K in Ks, i in 3:length(grid)-2                      # interior rows (away from clamped edges)
        @test sum(K[i, :]) ≈ 1
        @test sum(K[i, :] .* grid) ≈ grid[i]               # mean-preserving
    end
    var(K, i) = sum(K[i, :] .* (grid .- grid[i]).^2)
    @test var(Ks[3], 6) > var(Ks[2], 6) > var(Ks[1], 6)    # dispersion grows with θ
end

@testset "scale — concave V insures (θ*=min), convex V gambles (θ*=max)" begin
    stg = ScaleChoiceStage(block; x_axis = :x, scale_axis = :scale,
                           dispersions = dispersions, shocks = shocks, weights = weights)

    concave = reshape(sqrt.(xs), :, 1)                     # √x — risk-averse
    Vc = vec(backward!(stg, concave, NamedTuple()))
    # The chosen value equals the MIN-dispersion continuation (no spread) at interior nodes.
    K0 = HouseholdStages._mps_kernel(xs, dispersions[1], shocks, weights)
    @test Vc[3:end-2] ≈ (K0 * sqrt.(xs))[3:end-2]

    convex = reshape(xs.^2, :, 1)                          # x² — risk-loving
    Vx = vec(backward!(stg, convex, NamedTuple()))
    Kmax = HouseholdStages._mps_kernel(xs, dispersions[end], shocks, weights)
    @test Vx[3:end-2] ≈ (Kmax * (xs.^2))[3:end-2]
end
