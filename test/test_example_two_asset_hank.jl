using Test
using HouseholdStages

# Two-asset HANK (KMV) via the auxiliary-choice-axis pattern — existing stages only, no bespoke stage.
# (1) the illiquid block's deposit (the cross-axis flow) equals its brute reference to machine
# precision; (2) the full model solves to a stationary steady state.

module TwoAssetHankExampleTest
using Test, HouseholdStages, Random
include(joinpath(@__DIR__, "..", "examples", "two_asset_hank", "model.jl"))

@testset "two-asset HANK — illiquid deposit block == brute (machine precision)" begin
    Random.seed!(5)
    bgrid = collect(range(0.0, 8.0; length = 14)); agrid = collect(range(0.0, 8.0; length = 11))
    nb, na = length(bgrid), length(agrid); r_a = 0.05; κ = 0.08
    χ(d) = κ * d^2; dep(ac, a) = agrid[ac] - (1 + r_a) * a
    full = GriddedLayout(:liquid => GriddedContinuous(bgrid), :illiquid => GriddedContinuous(agrid),
                         :illiquid_choice => Discrete(collect(1:na)))
    choose = ArgmaxStage(full; axis = :illiquid_choice, reward = zeros(na, 1))
    debit  = WealthChangeStage(full; axis = :liquid,
               wealth_post = (; illiquid_choice, illiquid, liquid) -> (d = dep(Int(illiquid_choice), illiquid); max(liquid - d - χ(d), 0.0)))
    credit = WealthChangeStage(full; axis = :illiquid, wealth_post = (; illiquid_choice) -> agrid[Int(illiquid_choice)])
    forget = ForgetfulSumStage(full; axis = :illiquid_choice)
    block  = choose ∘ debit ∘ credit ∘ forget

    Vnext = randn(nb, na)
    Vs    = reshape(backward!(block, reshape(Vnext, nb, na, 1), NamedTuple()), nb, na)
    # The post-deposit liquid `bp = b − d − χ(d)` can land off-grid-RIGHT: withdrawing from illiquid
    # (d < 0) ADDS to liquid, so the TOP liquid cell (b = b_max) maps to b_max + (net withdrawal
    # benefit) — a slope-1 map with a fixed positive additive term (≈ +3.12 here, the max of −d−χ(d)).
    # No widening of b_max can clear that (bp_max = b_max + 3.12 > b_max for any b_max), so this is a
    # genuine grid-top corner where the off-grid CLAMP is the correct economics. The brute reference
    # therefore clamps right too (snap to `Va[end]`), apples-to-apples with the InterpKernel backward.
    interp_b(Va, bp) = (bp <= bgrid[1] ? Va[1] : bp >= bgrid[end] ? Va[end] : begin
        j = min(searchsortedlast(bgrid, bp), nb - 1); t = (bp - bgrid[j]) / (bgrid[j+1] - bgrid[j])
        (1 - t) * Va[j] + t * Va[j+1] end)
    brute = fill(-Inf, nb, na)
    for i in 1:nb, k in 1:na, ac in 1:na
        d = dep(ac, agrid[k]); bp = max(bgrid[i] - d - χ(d), 0.0)
        v = interp_b(@view(Vnext[:, ac]), bp); v > brute[i, k] && (brute[i, k] = v)
    end
    @test Vs ≈ brute
end

@testset "two-asset HANK — full model solves end-to-end" begin
    hh  = two_asset_household(TwoAssetParams(N_b = 18, N_a = 9))
    res = solve_steady_state_given_env!(hh, NamedTuple())
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-5)
    @test all(isfinite, res.V)
    m = compute_moments(hh, res.Λ, NamedTuple())
    @test m.mean_liquid > 0
    @test m.mean_illiquid > 0
end
end
