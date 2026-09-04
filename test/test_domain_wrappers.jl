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
                                   service = serv, adjustment_cost = adj)
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

# The layer split — which domain sugar is crosswalk-capable, and which keeps one gridding #
#----------------------------------------------------------------------------------------#
# A sugar constructor takes two layouts exactly when its payoff is arithmetic over the two
# operative coordinates, or is a cost matrix indexed by level. It takes one when the payoff is a
# level identity or an index lookup, which needs both ends drawn from the same set. The tests below
# state both halves: the two-layout forms are exercised at a genuine crosswalk, and the restricted
# six are pinned as taking one layout.

@testset "Two-layout sugar — arithmetic payoffs span a crosswalk" begin
    # 4 origin nodes → 3 destination nodes, at different coordinates.
    g4, g3 = [1.0, 2.0, 3.0, 4.0], [0.25, 1.5, 2.75]
    L4 = GriddedLayout(:wealth => GriddedContinuous(g4))
    L3 = GriddedLayout(:wealth => GriddedContinuous(g3))

    move = WealthChangeStage(L4, L3; wealth_post = (; wealth) -> 0.5 * wealth + 0.2)
    @test start_layout(move) == L4 && end_layout(move) == L3
    @test size(backward!(move, [1.0, 2.0, 3.0], NamedTuple())) == (4,)
    @test sum(forward!(move, fill(0.25, 4))) ≈ 1.0                    # mass conserved 4 → 3

    save = ConsumptionSavingsStage(L4, L3; β = 0.95, utility = (cell, c) -> log(c), axis = :wealth)
    # `c = b − a'` is arithmetic over the two grids: `b` on L4, `a'` on L3. The value is the NODE
    # max (only the policy is off-grid), so the closed form sweeps the two grids independently.
    V_end = [0.1, 0.2, 0.3]
    V = backward!(save, V_end, NamedTuple())
    @test size(V) == (4,)
    @test V ≈ [maximum(log(b - g3[k]) + 0.95 * V_end[k] for k in 1:3 if b > g3[k]) for b in g4] atol = 1e-12
    @test sum(forward!(save, fill(0.25, 4))) ≈ 1.0

    Lh4 = GriddedLayout(:h => GriddedContinuous([1.0, 2.0, 3.0, 4.0]))
    Lh3 = GriddedLayout(:h => GriddedContinuous([1.5, 2.5, 3.5]))
    invest = CapitalInvestmentStage(Lh4, Lh3; axis = :h, β = 0.9,
                                    production = (h; env) -> 2.0 * h, effort_cost = (i; env) -> i^2)
    @test size(backward!(invest, [0.0, 1.0, 2.0], NamedTuple())) == (4,)
    @test sum(forward!(invest, fill(0.25, 4))) ≈ 1.0

    Ld4 = GriddedLayout(:durable => GriddedContinuous([0.0, 1.0, 2.0, 3.0]))
    Ld3 = GriddedLayout(:durable => GriddedContinuous([0.5, 1.5, 2.5]))
    dur = DurableAdjustmentStage(Ld4, Ld3; axis = :durable, β = 1.0,
                                 service = (d; env) -> 3.0 * d, adjustment_cost = (Δ; env) -> Δ^2)
    @test size(backward!(dur, [0.0, 0.5, 1.0], NamedTuple())) == (4,)

    Lw4 = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]), :income => Discrete([0.5, 1.5]))
    Lw3 = GriddedLayout(:wealth => GriddedContinuous([0.25, 1.5, 2.75]), :income => Discrete([0.5, 1.5]))
    inc = IncomeStage(Lw4, Lw3; axis = :wealth)
    @test size(backward!(inc, ones(3, 2), (r = 0.02, w = 1.0))) == (4, 2)

    Ls4 = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]), :sh => GriddedContinuous([0.0, 1.0]))
    Ls3 = GriddedLayout(:wealth => GriddedContinuous([0.25, 1.5, 2.75]), :sh => GriddedContinuous([0.0, 1.0]))
    reval = AssetPriceChangeStage(Ls4, Ls3; holdings_axis = :sh)
    @test size(backward!(reval, ones(3, 2), (q = 1.1, q_last = 1.0))) == (4, 2)

    # And each one-layout form is still sugar for the equal pair.
    square = WealthChangeStage(L4; wealth_post = (; wealth) -> wealth)
    @test start_layout(square) == L4 && end_layout(square) == L4
end

@testset "Two-layout sugar — a cost matrix over levels spans a rectangle" begin
    # The cost is indexed by level, so the two ends may carry different choice sets entirely.
    Lo = GriddedLayout(:loc => Discrete([:a, :b, :c]))
    Ld = GriddedLayout(:loc => Discrete([:x, :y]))
    C  = [0.0 0.5; 0.4 0.0; 0.9 0.2]                      # (n_origin, n_destination)
    V_end = [1.0, 2.0]
    ε = 1.0

    for stage in (MigrationStage(Lo, Ld; axis = :loc, migration_cost = C, ε = ε),
                  SectorSwitchingStage(Lo, Ld; axis = :loc, switching_cost = C, ε = ε),
                  DirectedSearchStage(Lo, Ld; axis = :loc, search_cost = C, ε = ε))
        @test start_layout(stage) == Lo && end_layout(stage) == Ld
        V = copy(backward!(stage, V_end, NamedTuple()))
        @test V ≈ [ε * log(sum(exp((-C[i, j] + V_end[j]) / ε) for j in 1:2)) for i in 1:3] atol = 1e-12
        P = choice_probabilities(stage)
        @test size(P) == (3, 2) && all(≈(1.0), sum(P; dims = 2))
        @test sum(forward!(stage, fill(1 / 3, 3))) ≈ 1.0 atol = 1e-12
    end

    # The destination payoff of `LogitUtilityStage` sits square on the END layout.
    u = [0.3, 0.0]
    chain = LogitUtilityStage(Lo, Ld; axis = :loc, cost_matrix = C,
                              utility = (; loc) -> loc == :x ? u[1] : u[2], ε = ε)
    V = copy(backward!(chain, V_end, NamedTuple()))
    @test V ≈ [ε * log(sum(exp((-C[i, j] + u[j] + V_end[j]) / ε) for j in 1:2)) for i in 1:3] atol = 1e-12
end

@testset "One-gridding sugar — the six restrictions are decisions, not omissions" begin
    # Each of these compares levels across the transition or indexes one shared grid, so its two
    # ends must be the same gridding; the reason is stated at each constructor's docstring.
    for sugar in (BuyHomeStage, SellHomeStage, DefaultStage,
                  AdvanceAgeStage, SearchMatchingStage, RetentionStage)
        @test hasmethod(sugar, Tuple{GriddedLayout})
        @test !hasmethod(sugar, Tuple{GriddedLayout, GriddedLayout})
    end
end
