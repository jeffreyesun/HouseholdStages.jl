using Test
using HouseholdStages
using LinearAlgebra

# Part-2 domain wrappers: recognizable model-named stages, each thin sugar over an EXISTING
# primitive (no new contraction). Each test pins the wrapper against the primitive it delegates to.

@testset "BenPorath — human-capital investment = ContinuousArgmax with effort-cost reward" begin
    hs    = [1.0, 2.0, 3.0, 4.0]
    layout = GriddedLayout(:h => GriddedContinuous(hs))
    prod   = (h; env) -> 2.0 * h                       # linear production
    effort = (i; env) -> i^2                            # convex effort cost
    stage  = CapitalInvestmentStage(layout; axis = :h, β = 0.9, production = prod, effort_cost = effort)
    V_end  = [0.0, 1.0, 2.0, 3.0]
    V      = backward!(stage, V_end, NamedTuple())
    # Direct max: V[h] = max_{h'} prod(h) − effort(h' − h) + β V_end[h'].
    expected = [maximum(prod(hs[b]; env=nothing) - effort(max(hs[a] - hs[b], 0.0); env=nothing) + 0.9 * V_end[a]
                        for a in 1:4) for b in 1:4]
    @test V ≈ expected
end

@testset "DurableAdjustment — service net of convex adjustment cost" begin
    ds    = [0.0, 1.0, 2.0, 3.0]
    layout = GriddedLayout(:durable => GriddedContinuous(ds))
    serv  = (d; env) -> 3.0 * d
    adj   = (Δ; env) -> Δ^2
    stage = DurableAdjustmentStage(layout; axis = :durable, β = 1.0,
                                   service = serv, adjustment_cost = adj, assume_monotone = true)
    V_end = [0.0, 0.5, 1.0, 1.5]
    V     = backward!(stage, V_end, NamedTuple())
    expected = [maximum(serv(ds[a]; env=nothing) - adj(ds[a] - ds[b]; env=nothing) + V_end[a] for a in 1:4) for b in 1:4]
    @test V ≈ expected
end

@testset "Default — gated {repay, default} argmax with penalty" begin
    layout = GriddedLayout(:status => Discrete([1, 2]))   # 1 = repay, 2 = default
    stage  = DefaultStage(layout; axis = :status, default_index = 2, default_penalty = 0.5)
    V_end  = [3.0, 4.0]                                  # repay-branch value 3, default-branch value 4
    V      = backward!(stage, V_end, NamedTuple())
    # Each origin picks max(repay: 0 + 3, default: −0.5 + 4) = max(3, 3.5) = 3.5.
    @test all(V .≈ 3.5)
end

@testset "DirectedSearch — logit over submarkets = LogitChoiceStage" begin
    layout = GriddedLayout(:submarket => Discrete([1, 2]))
    C      = [0.0 0.3; 0.3 0.0]
    ds     = DirectedSearchStage(layout; axis = :submarket, search_cost = C, ε = 0.5)
    ref    = LogitChoiceStage(layout; axis = :submarket, cost_matrix = C, ε = 0.5)
    V_end  = [1.0, 2.0]
    @test backward!(ds, V_end, NamedTuple()) ≈ backward!(ref, V_end, NamedTuple())
end

# (RI-discrete is intentionally NOT a wrapper — it is the composition tested in
# test_rational_inattention.jl; LogitUtilityStage already serves.)
