using Test
using HouseholdStages

# LogitUtilityStage = LogitChoiceStage ∘ UtilityStage: the general
# state-dependent discrete choice. The destination payoff u(cell) enters V
# through the UtilityStage; only the origin-dependent cost lives on the logit.

@testset "LogitUtilityStage — reproduces the state-dependent logit closed form" begin
    layout = GriddedLayout(
        StateAxis(:income,   discrete_finite([0.6, 1.4])),
        StateAxis(:location, categorical([:A, :B, :C])),
    )
    ε = 0.8
    C = [0.0 0.5 0.7;
         0.5 0.0 0.5;
         0.7 0.5 0.0]
    dest_value(cell; env) = cell.location === :B ? 0.8 * cell.income :
                            cell.location === :C ? 0.6 / cell.income : 0.0

    stage = LogitUtilityStage(layout;
        choice_axis = :location, cost_matrix = C, utility = dest_value, ε = ε)
    @test stage isa ChainStage

    V_end = Float64[0.3 * zi + 0.1 * li for zi in 1:2, li in 1:3]
    V_pre = backward!(stage, V_end, NamedTuple())

    z_grid = [0.6, 1.4]; locs = [:A, :B, :C]
    u(j, z) = locs[j] === :B ? 0.8 * z : locs[j] === :C ? 0.6 / z : 0.0
    expected = [ε * log(sum(exp((-C[i, j] + u(j, z_grid[zi]) + V_end[zi, j]) / ε) for j in 1:3))
                for zi in 1:2, i in 1:3]
    @test isapprox(V_pre, expected; atol = 1e-12)
end

@testset "LogitUtilityStage — equals the manual LogitChoiceStage ∘ UtilityStage" begin
    layout = GriddedLayout(
        StateAxis(:income,   discrete_finite([0.6, 1.4])),
        StateAxis(:location, categorical([:A, :B, :C])),
    )
    ε = 0.7
    C = [0.0 0.4 0.9; 0.4 0.0 0.4; 0.9 0.4 0.0]
    payoff(cell; env) = cell.location === :C ? 0.5 : 0.0

    packaged = LogitUtilityStage(layout;
        choice_axis = :location, cost_matrix = C, utility = payoff, ε = ε)
    manual = LogitChoiceStage(layout; choice_axis = :location, cost_matrix = C, ε = ε) ∘
             UtilityStage(layout; utility = payoff)

    V_end = randn(2, 3)
    @test backward!(packaged, copy(V_end), NamedTuple()) ==
          backward!(manual, copy(V_end), NamedTuple())
end
