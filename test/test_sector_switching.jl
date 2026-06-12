using Test
using HouseholdStages

# SectorSwitchingStage is sugar over LogitChoiceStage on a sector axis — these
# tests confirm the wrapper wires through to the same transition-cost logit.

@testset "SectorSwitchingStage — backward matches the logit closed form" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid(collect(range(0.0, 1.0; length = 3)))),
        StateAxis(:sector, categorical([:ag, :mfg, :svc])),
    )
    C = [0.0 0.4 0.6;
         0.4 0.0 0.4;
         0.6 0.4 0.0]
    ε = 0.7
    stage = SectorSwitchingStage(layout; sector_axis = :sector, switching_cost = C, ε = ε)

    n_w, n_s = axissize.(layout.axes)
    V_end = Float64[0.1 * w + 0.2 * s for w in 1:n_w, s in 1:n_s]
    V_pre = copy(backward!(stage, V_end, NamedTuple()))
    for w in 1:n_w, i in 1:n_s
        expected = ε * log(sum(exp((-C[i, j] + V_end[w, j]) / ε) for j in 1:n_s))
        @test V_pre[w, i] ≈ expected atol = 1e-12
    end

    Λ_start = fill(1.0 / (n_w * n_s), n_w, n_s)
    @test sum(forward!(stage, Λ_start)) ≈ 1.0 atol = 1e-12
end

@testset "SectorSwitchingStage — equals MigrationStage with the same cost" begin
    # Same primitive, different axis name: identical numerics.
    layout_s = GriddedLayout(StateAxis(:wealth, continuous_grid([0.0, 1.0])),
                           StateAxis(:sector, categorical([:a, :b])))
    layout_l = GriddedLayout(StateAxis(:wealth, continuous_grid([0.0, 1.0])),
                           StateAxis(:location, categorical([:a, :b])))
    C = [0.0 0.5; 0.5 0.0]
    sec = SectorSwitchingStage(layout_s; sector_axis = :sector, switching_cost = C, ε = 1.3)
    mig = MigrationStage(layout_l; location_axis = :location, migration_cost = C, ε = 1.3)
    V = randn(2, 2)
    @test backward!(sec, V, NamedTuple()) ≈ backward!(mig, V, NamedTuple()) atol = 1e-14
end
