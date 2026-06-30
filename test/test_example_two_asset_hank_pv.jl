using Test
using HouseholdStages

# Two-asset HANK (KMV) via Route A′ — the PORTFOLIO-VALUE reformulation (state (W, a), liquid = W − a),
# existing stages only, NO auxiliary axis. Two tests:
#   (1) the novel Rebalance ∘ Consume block's backward value equals its brute (W, a) Bellman to machine
#       precision — both argmaxes choose grid indices and read continuations on-grid, so the block is
#       exact (no interpolation to match);
#   (2) the full model solves to a stationary steady state with sensible moments.

module TwoAssetHankPvExampleTest
using Test, HouseholdStages, Random
include(joinpath(@__DIR__, "..", "examples", "two_asset_hank_pv", "model.jl"))

# (1) Proof: build Rebalance ∘ Consume alone (one income state) and check it equals
#     V(W, a) = max_{a'} [ −χ(a'−a) + max_{W' ≥ a', W' ≤ W} ( u(W−W') + β·V_next(W', a') ) ].
@testset "two-asset HANK Route A′ — Rebalance∘Consume == brute Bellman (machine precision)" begin
    Random.seed!(11)
    p = TwoAssetPvParams(N_W = 20, N_a = 9, β = 0.95, κ = 0.08, σ = 2.0,
                         W_min = 0.1, W_max = 12.0, a_max = 8.0)
    Wgrid = collect(range(p.W_min, p.W_max; length = p.N_W))
    agrid = collect(range(0.0, p.a_max; length = p.N_a))
    nW, na = p.N_W, p.N_a
    χ(d) = p.κ * d^2

    layout = GriddedLayout(:wealth => GriddedContinuous(Wgrid),
                           :illiquid => GriddedContinuous(agrid))

    rebalance = ArgmaxStage(layout; axis = :illiquid,
        reward = [-χ(agrid[ap] - agrid[a]) for ap in 1:na, a in 1:na])
    consume   = ArgmaxStage(layout; axis = :wealth,
        reward = _PvConsumeReward(Wgrid, p), search = :divide_conquer, assume_monotone = true) ∘
        TimeDiscountingStage(layout; β = p.β)
    block = rebalance ∘ consume

    Vnext   = randn(nW, na)
    V_start = copy(backward!(block, Vnext, NamedTuple()))

    # Brute: Consume (on-grid W'), then Rebalance (on-grid a'). No interpolation anywhere. Feasibility
    # and the ε floor mirror _PvConsumeReward exactly.
    Vc = fill(-Inf, nW, na)
    for k in 1:na, i in 1:nW                       # state (W = Wgrid[i], a = agrid[k])
        for j in 1:nW                              # choice W' = Wgrid[j]
            (Wgrid[j] >= agrid[k] && Wgrid[j] <= Wgrid[i]) || continue   # liquid ≥ 0 & W' ≤ W
            c = max(Wgrid[i] - Wgrid[j], p.ε)
            v = u_crra(c, Val(p.σ)) + p.β * Vnext[j, k]
            v > Vc[i, k] && (Vc[i, k] = v)
        end
    end
    brute = fill(-Inf, nW, na)
    for k in 1:na, i in 1:nW                       # state (W, a = agrid[k])
        for kp in 1:na                             # choice a' = agrid[kp]
            v = -χ(agrid[kp] - agrid[k]) + Vc[i, kp]
            v > brute[i, k] && (brute[i, k] = v)
        end
    end

    fin = isfinite.(brute)
    @test any(fin)
    @test V_start[fin] ≈ brute[fin]
    # Where the brute frontier is empty (no feasible (a', W')), the chain also yields −Inf.
    @test all(V_start[.!fin] .== -Inf)
end

# (2) Full income-fluctuation model solves to a stationary steady state, no auxiliary axis.
@testset "two-asset HANK Route A′ — full model solves end-to-end" begin
    hh  = two_asset_pv_household(TwoAssetPvParams(N_W = 24, N_a = 9))
    res = solve_steady_state_given_env!(hh, NamedTuple())
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-5)
    @test all(isfinite, res.V)
    m = compute_moments(hh, res.Λ, NamedTuple())
    @test m.mean_liquid  > 0
    @test m.mean_illiquid > 0
    @test 0.0 < m.mean_illiquid / (m.mean_liquid + m.mean_illiquid) < 1.0
end
end
