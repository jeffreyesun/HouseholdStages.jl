using Test
using HouseholdStages

# Becker–Murphy rational addiction via the AUXILIARY-CHOICE-AXIS pattern — existing stages only, no
# bespoke household stage. Two tests: (1) the pattern's backward value equals the brute habit Bellman
# to machine precision (the proof the construction is correct); (2) the full income-fluctuation model
# solves end-to-end through the standard outer loop.

module HabitExampleTest
using Test, HouseholdStages, Random
include(joinpath(@__DIR__, "..", "examples", "habit", "model.jl"))

# (1) Proof: build the choice block alone (no income) and check it equals
#     V(x,S) = max_{b'} u(x−b',S) + β·V_next(b', (1−δ_S)S + (x−b')).
@testset "habit — auxiliary-axis chain == brute Bellman (machine precision)" begin
    Random.seed!(7)
    p = HabitParams(N_w = 16, β = 0.95, δ_S = 0.4)
    wgrid = collect(range(0.0, p.w_max; length = p.N_w))
    hgrid = collect(range(0.0, p.S_max; length = p.N_S)); nw, nh = p.N_w, p.N_S
    full = GriddedLayout(:wealth => GriddedContinuous(wgrid),
                         :habit => GriddedContinuous(hgrid),
                         :savings_choice => Discrete(collect(1:nw)))
    choose   = ArgmaxStage(full; axis = :savings_choice, reward = zeros(nw, 1))
    felicity = UtilityStage(full; utility = (; wealth, savings_choice, habit) -> u_habit(wealth - wgrid[Int(savings_choice)], habit, p))
    disc     = TimeDiscountingStage(full; β = p.β)
    habitup  = WealthChangeStage(full; axis = :habit,
                 wealth_post = (; habit, wealth, savings_choice) -> (1 - p.δ_S) * habit + (wealth - wgrid[Int(savings_choice)]))
    setliq   = WealthChangeStage(full; axis = :wealth, wealth_post = (; savings_choice) -> wgrid[Int(savings_choice)])
    forget   = ForgetfulSumStage(full; axis = :savings_choice)
    block    = choose ∘ felicity ∘ disc ∘ habitup ∘ setliq ∘ forget

    Vnext   = randn(nw, nh)
    V_start = reshape(backward!(block, reshape(Vnext, nw, nh, 1), NamedTuple()), nw, nh)

    interp_h(Vw, Sp) = (Sp <= hgrid[1] ? Vw[1] : begin
        j = min(searchsortedlast(hgrid, Sp), nh - 1); t = (Sp - hgrid[j]) / (hgrid[j+1] - hgrid[j])
        (1 - t) * Vw[j] + t * Vw[j+1] end)
    brute = fill(-Inf, nw, nh)
    for i in 1:nw, k in 1:nh, jb in 1:nw
        c = wgrid[i] - wgrid[jb]; c < 0 && continue
        val = u_habit(c, hgrid[k], p) + p.β * interp_h(@view(Vnext[jb, :]), (1 - p.δ_S) * hgrid[k] + c)
        val > brute[i, k] && (brute[i, k] = val)
    end
    fin = isfinite.(brute)
    @test V_start[fin] ≈ brute[fin]
end

# (2) Full income-fluctuation model solves to a stationary steady state.
@testset "habit — full model solves end-to-end" begin
    hh  = habit_household(HabitParams(N_w = 20, N_S = 10))
    res = solve_steady_state_given_env!(hh, NamedTuple())
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-5)
    @test all(isfinite, res.V)
    m = compute_moments(hh, res.Λ, NamedTuple())
    @test m.mean_wealth > 0
    @test m.mean_habit  > 0
end
end
